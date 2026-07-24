## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.001618947


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0031278, 0.0031278)
1: (-0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010969, 0.0010969)
2: (0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041736, 0.0041736)
3: (1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335)
4: (-0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006821, 0.0006821)
5: (0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0024083, 0.0024083)
6: (-0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811)
7: (-0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043632, 0.0043632)
8: (-0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0073924, 0.0073924)
9: (-0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0036639, 0.0036639)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 2.33 = 3.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0019143, upper bound: 0.0019143

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018362, upper bound: 0.0018362
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018362, upper bound: 0.0018362
time: 1.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.20 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.20
Output dim: 3, lower bound: -0.0018362, upper bound: 0.0018362
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.20
Output dim: 3, lower bound: -0.0018362, upper bound: 0.0018362

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0031224, 0.0031284
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010944, 0.0010973
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041655, 0.0041746
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006823, 0.0006806
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0024041, 0.0024088
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043632, 0.0043625
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0073956, 0.0073759
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0036550, 0.0036655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017826, upper bound: 0.0017826
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017826, upper bound: 0.0017826
time: 1.14 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0031278, 0.0031224
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010969, 0.0010944
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041736, 0.0041655
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006806, 0.0006821
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0024083, 0.0024041
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043625, 0.0043632
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0073759, 0.0073924
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0036639, 0.0036550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017826, upper bound: 0.0017826
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017826, upper bound: 0.0017826
time: 1.16 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 3, lower bound: -0.0017826, upper bound: 0.0017826
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 3, lower bound: -0.0017826, upper bound: 0.0017826
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 3, lower bound: -0.0017826, upper bound: 0.0017826
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 3, lower bound: -0.0017826, upper bound: 0.0017826

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030903, 0.0031027
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010781, 0.0010842
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041137, 0.0041328
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006746, 0.0006711
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023783, 0.0023881
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043600, 0.0043586
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0073062, 0.0072648
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035958, 0.0036179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017623, upper bound: 0.0017772
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017772, upper bound: 0.0017623
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0031224, 0.0030962
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010944, 0.0010810
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041655, 0.0041228
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006728, 0.0006806
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0024041, 0.0023830
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043593, 0.0043625
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0072846, 0.0073759
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0036550, 0.0036063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017623, upper bound: 0.0017772
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017772, upper bound: 0.0017623
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030957, 0.0030965
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010806, 0.0010812
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041218, 0.0041233
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006728, 0.0006725
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023825, 0.0023832
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043593, 0.0043593
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0072856, 0.0072815
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0036047, 0.0036069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017623, upper bound: 0.0017772
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017772, upper bound: 0.0017623
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0031278, 0.0030903
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010969, 0.0010781
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041736, 0.0041137
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006711, 0.0006821
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0024083, 0.0023783
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043586, 0.0043632
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0072648, 0.0073924
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0036639, 0.0035958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017623, upper bound: 0.0017772
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017772, upper bound: 0.0017623
time: 1.03 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.60 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 3, lower bound: -0.0017623, upper bound: 0.0017772
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 3, lower bound: -0.0017772, upper bound: 0.0017623
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 3, lower bound: -0.0017623, upper bound: 0.0017772
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 3, lower bound: -0.0017772, upper bound: 0.0017623
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 3, lower bound: -0.0017623, upper bound: 0.0017772
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 3, lower bound: -0.0017772, upper bound: 0.0017623
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 3, lower bound: -0.0017623, upper bound: 0.0017772
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 3, lower bound: -0.0017772, upper bound: 0.0017623

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030773, 0.0030824
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010724, 0.0010750
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040937, 0.0041017
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006688, 0.0006673
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023681, 0.0023721
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043577, 0.0043571
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0072387, 0.0072215
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035726, 0.0035818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030700, 0.0030893
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010689, 0.0010783
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040826, 0.0041123
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006708, 0.0006653
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023624, 0.0023776
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043585, 0.0043562
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0072617, 0.0071974
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035597, 0.0035941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0031094, 0.0030759
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010887, 0.0010718
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041455, 0.0040917
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006670, 0.0006769
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023939, 0.0023670
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043569, 0.0043610
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0072171, 0.0073326
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0036318, 0.0035703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0031022, 0.0030828
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010852, 0.0010751
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041343, 0.0041023
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006689, 0.0006748
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023882, 0.0023724
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043577, 0.0043601
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0072400, 0.0073084
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0036189, 0.0035825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030827, 0.0030763
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010749, 0.0010719
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041018, 0.0040922
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006670, 0.0006688
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023723, 0.0023673
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043569, 0.0043578
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0072182, 0.0072382
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035815, 0.0035708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030754, 0.0030837
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010713, 0.0010756
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040907, 0.0041035
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006692, 0.0006667
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023666, 0.0023731
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043578, 0.0043569
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0072428, 0.0072141
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035686, 0.0035840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0031148, 0.0030700
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010912, 0.0010689
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041536, 0.0040826
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006653, 0.0006784
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023981, 0.0023624
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043562, 0.0043617
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071974, 0.0073491
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0036408, 0.0035597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0031076, 0.0030773
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010876, 0.0010724
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041424, 0.0040937
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006673, 0.0006763
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023924, 0.0023681
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043571, 0.0043608
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0072215, 0.0073250
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0036279, 0.0035726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
time: 1.24 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017436, upper bound: 0.0017565
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 3, lower bound: -0.0017565, upper bound: 0.0017436

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030431, 0.0030478
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010558, 0.0010581
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040415, 0.0040488
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006590, 0.0006577
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023412, 0.0023449
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043533, 0.0043528
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071244, 0.0071087
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035121, 0.0035205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017085
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030435, 0.0030482
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010560, 0.0010583
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040422, 0.0040495
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006591, 0.0006578
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023416, 0.0023453
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043534, 0.0043528
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071259, 0.0071101
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035129, 0.0035213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017086
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030358, 0.0030540
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010523, 0.0010612
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040304, 0.0040584
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006608, 0.0006556
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023355, 0.0023499
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043541, 0.0043519
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071453, 0.0070846
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0034992, 0.0035317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017086, upper bound: 0.0017027
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030367, 0.0030551
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010527, 0.0010617
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040318, 0.0040601
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006611, 0.0006559
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023363, 0.0023507
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043542, 0.0043520
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071489, 0.0070876
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035009, 0.0035336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017085, upper bound: 0.0017027
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030748, 0.0030404
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010718, 0.0010545
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040927, 0.0040375
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006569, 0.0006672
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023666, 0.0023392
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043525, 0.0043567
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0070999, 0.0072210
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035706, 0.0035074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017085
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030752, 0.0030418
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010720, 0.0010551
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040933, 0.0040395
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006573, 0.0006674
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023669, 0.0023402
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043526, 0.0043567
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071043, 0.0072224
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035713, 0.0035098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017086
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030675, 0.0030472
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010683, 0.0010578
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040816, 0.0040479
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006588, 0.0006652
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023609, 0.0023445
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043533, 0.0043558
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071224, 0.0071969
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035577, 0.0035194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017086, upper bound: 0.0017027
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030685, 0.0030486
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010687, 0.0010585
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040830, 0.0040501
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006593, 0.0006654
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023616, 0.0023456
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043534, 0.0043559
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071272, 0.0071999
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035594, 0.0035220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017085, upper bound: 0.0017027
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030483, 0.0030427
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010583, 0.0010556
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040497, 0.0040410
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006576, 0.0006591
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023454, 0.0023410
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043527, 0.0043535
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071075, 0.0071258
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035213, 0.0035115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017085
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030488, 0.0030421
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010585, 0.0010553
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040504, 0.0040400
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006574, 0.0006592
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023457, 0.0023404
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043527, 0.0043535
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071053, 0.0071271
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035220, 0.0035103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017086
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030411, 0.0030501
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010548, 0.0010592
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040386, 0.0040524
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006597, 0.0006570
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023397, 0.0023468
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043536, 0.0043526
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071321, 0.0071016
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035084, 0.0035247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017086, upper bound: 0.0017027
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030420, 0.0030495
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010552, 0.0010589
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040400, 0.0040514
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006595, 0.0006573
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023404, 0.0023463
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043535, 0.0043527
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071300, 0.0071047
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035100, 0.0035235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017085, upper bound: 0.0017027
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030802, 0.0030367
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010744, 0.0010527
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041008, 0.0040318
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006559, 0.0006687
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023708, 0.0023363
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043520, 0.0043574
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0070876, 0.0072380
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035800, 0.0035009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017085
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030806, 0.0030358
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010746, 0.0010523
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0041014, 0.0040304
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006556, 0.0006688
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023711, 0.0023355
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043519, 0.0043574
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0070846, 0.0072394
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035807, 0.0034992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017086
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030729, 0.0030435
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010709, 0.0010560
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040896, 0.0040422
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006578, 0.0006666
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023651, 0.0023416
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043528, 0.0043565
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071101, 0.0072139
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035671, 0.0035129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017086, upper bound: 0.0017027
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0030739, 0.0030431
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010713, 0.0010558
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0040910, 0.0040415
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006577, 0.0006669
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0023658, 0.0023412
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043528, 0.0043566
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0071087, 0.0072170
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0035687, 0.0035121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017085, upper bound: 0.0017027
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
time: 1.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017085
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017086
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017086, upper bound: 0.0017027
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017085, upper bound: 0.0017027
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017085
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017086
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017086, upper bound: 0.0017027
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017085, upper bound: 0.0017027
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017085
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017086
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017086, upper bound: 0.0017027
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017085, upper bound: 0.0017027
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017085
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0016930, upper bound: 0.0017150
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017027, upper bound: 0.0017086
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017086, upper bound: 0.0017027
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017085, upper bound: 0.0017027
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.02
Output dim: 3, lower bound: -0.0017150, upper bound: 0.0016930

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029419, 0.0029375
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010042, 0.0010021
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038781, 0.0038713
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006254, 0.0006267
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022611, 0.0022576
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043429, 0.0043434
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067324, 0.0067471
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033155, 0.0033076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029327, 0.0029486
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0009998, 0.0010075
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038640, 0.0038885
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006286, 0.0006241
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022539, 0.0022664
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043442, 0.0043423
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067696, 0.0067167
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0032992, 0.0033275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029425, 0.0029379
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010045, 0.0010023
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038790, 0.0038720
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006256, 0.0006269
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022616, 0.0022580
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043429, 0.0043435
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067339, 0.0067492
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033166, 0.0033085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029332, 0.0029477
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010000, 0.0010071
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038647, 0.0038870
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006284, 0.0006242
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022543, 0.0022657
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043441, 0.0043424
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067664, 0.0067180
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033000, 0.0033258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029357, 0.0029437
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010012, 0.0010051
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038686, 0.0038809
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006272, 0.0006249
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022563, 0.0022626
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043436, 0.0043427
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067532, 0.0067265
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033045, 0.0033188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029255, 0.0029536
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0009962, 0.0010100
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038529, 0.0038961
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030327
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006301, 0.0006220
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022482, 0.0022704
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043448, 0.0043415
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067862, 0.0066925
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0032864, 0.0033364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029364, 0.0029448
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010015, 0.0010057
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038696, 0.0038826
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006276, 0.0006251
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022568, 0.0022634
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043438, 0.0043428
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067569, 0.0067288
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033058, 0.0033207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029264, 0.0029536
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0009967, 0.0010099
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038543, 0.0038960
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006301, 0.0006223
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022490, 0.0022703
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043448, 0.0043416
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067860, 0.0066956
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0032880, 0.0033363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029741, 0.0029301
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010208, 0.0009985
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039299, 0.0038600
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006233, 0.0006364
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022866, 0.0022519
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043420, 0.0043473
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067079, 0.0068603
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033757, 0.0032946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029650, 0.0029413
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010164, 0.0010039
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039158, 0.0038771
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006265, 0.0006337
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022794, 0.0022606
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043433, 0.0043462
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067450, 0.0068298
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033594, 0.0033144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029748, 0.0029314
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010211, 0.0009991
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039308, 0.0038620
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006237, 0.0006365
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022871, 0.0022529
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043422, 0.0043474
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067123, 0.0068624
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033768, 0.0032969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029654, 0.0029408
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010166, 0.0010037
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039165, 0.0038765
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006264, 0.0006339
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022797, 0.0022603
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043433, 0.0043463
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067436, 0.0068312
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033602, 0.0033137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029680, 0.0029369
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010178, 0.0010018
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039204, 0.0038703
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006253, 0.0006346
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022817, 0.0022572
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043428, 0.0043466
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067303, 0.0068397
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033647, 0.0033066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029578, 0.0029471
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010128, 0.0010068
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039047, 0.0038861
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006282, 0.0006317
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022737, 0.0022652
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043440, 0.0043454
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067645, 0.0068057
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033465, 0.0033248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029687, 0.0029383
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010181, 0.0010025
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039215, 0.0038725
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006257, 0.0006348
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022823, 0.0022583
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043430, 0.0043466
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067351, 0.0068420
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033659, 0.0033091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029587, 0.0029463
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010133, 0.0010064
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039061, 0.0038849
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006280, 0.0006319
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022744, 0.0022646
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043439, 0.0043455
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067620, 0.0068088
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033482, 0.0033235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029472, 0.0029324
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010068, 0.0009996
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038861, 0.0038635
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006240, 0.0006281
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022653, 0.0022537
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043423, 0.0043441
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067155, 0.0067634
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033249, 0.0032986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029380, 0.0029444
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010023, 0.0010054
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038721, 0.0038819
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006274, 0.0006255
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022581, 0.0022631
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043437, 0.0043430
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067553, 0.0067329
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033087, 0.0033199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029478, 0.0029317
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010071, 0.0009993
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038871, 0.0038625
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006238, 0.0006283
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022658, 0.0022531
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043422, 0.0043442
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067133, 0.0067654
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033260, 0.0032975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029385, 0.0029427
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010025, 0.0010046
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038727, 0.0038794
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006270, 0.0006256
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022584, 0.0022618
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043435, 0.0043431
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067500, 0.0067343
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033094, 0.0033171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029410, 0.0029398
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010038, 0.0010032
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038766, 0.0038748
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006261, 0.0006263
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022604, 0.0022595
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043432, 0.0043434
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067401, 0.0067428
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033139, 0.0033118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029308, 0.0029500
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0009988, 0.0010082
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038610, 0.0038905
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006290, 0.0006234
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022524, 0.0022675
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043444, 0.0043422
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067741, 0.0067088
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0032958, 0.0033299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029417, 0.0029391
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010041, 0.0010029
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038777, 0.0038738
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006259, 0.0006265
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022610, 0.0022590
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043431, 0.0043435
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067379, 0.0067451
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033152, 0.0033106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029317, 0.0029492
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0009992, 0.0010078
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0038624, 0.0038893
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006288, 0.0006237
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022531, 0.0022669
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043443, 0.0043423
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067713, 0.0067119
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0032974, 0.0033285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029795, 0.0029264
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010234, 0.0009967
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039379, 0.0038543
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006223, 0.0006378
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022908, 0.0022490
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043416, 0.0043480
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0066956, 0.0068764
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033849, 0.0032880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029703, 0.0029364
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010189, 0.0010015
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039238, 0.0038696
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006251, 0.0006352
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022836, 0.0022568
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043428, 0.0043469
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067288, 0.0068460
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033687, 0.0033058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029801, 0.0029255
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010237, 0.0009962
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039388, 0.0038529
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030327, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006220, 0.0006379
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022913, 0.0022482
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043415, 0.0043481
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0066925, 0.0068785
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033860, 0.0032864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029707, 0.0029357
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010191, 0.0010012
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039244, 0.0038686
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006249, 0.0006353
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022839, 0.0022563
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043427, 0.0043470
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067265, 0.0068474
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033694, 0.0033045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029733, 0.0029332
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010203, 0.0010000
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039284, 0.0038647
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006242, 0.0006360
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022859, 0.0022543
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043424, 0.0043473
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067180, 0.0068558
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033739, 0.0033000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029631, 0.0029425
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010153, 0.0010045
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039127, 0.0038790
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006269, 0.0006331
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022779, 0.0022616
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043435, 0.0043461
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067492, 0.0068219
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033558, 0.0033166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029740, 0.0029327
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010207, 0.0009998
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039294, 0.0038640
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006241, 0.0006362
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022864, 0.0022539
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043423, 0.0043473
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067167, 0.0068582
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033752, 0.0032992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027234, 0.0007870, -0.0027234, 0.0007870, -0.0029640, 0.0029419
1: -0.0045626, -0.0033799, -0.0045626, -0.0033799, -0.0010158, 0.0010042
2: 0.0111392, 0.0158875, 0.0111392, 0.0158875, -0.0039141, 0.0038781
3: 1.0068343, 1.0098678, 1.0068343, 1.0098678, -0.0030335, 0.0030335
4: -0.0042256, -0.0034395, -0.0042256, -0.0034395, -0.0006267, 0.0006333
5: 0.0018693, 0.0045775, 0.0018693, 0.0045775, -0.0022786, 0.0022611
6: -0.0025939, -0.0023127, -0.0025939, -0.0023127, -0.0002811, 0.0002811
7: -0.0130882, -0.0086822, -0.0130882, -0.0086822, -0.0043434, 0.0043462
8: -0.0133967, -0.0048173, -0.0133967, -0.0048173, -0.0067471, 0.0068249
9: -0.0018310, 0.0024552, -0.0018310, 0.0024552, -0.0033574, 0.0033155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
time: 0.88 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014441, upper bound: 0.0014535
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014468, upper bound: 0.0014500
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014500, upper bound: 0.0014468
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.64
Output dim: 3, lower bound: -0.0014535, upper bound: 0.0014441

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.79 + 325.12 = 328.91 seconds
