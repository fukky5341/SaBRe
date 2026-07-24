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
Threshold: 0.00379488


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0003077, 0.0000969, -0.0003077, 0.0000969, -0.0002265, 0.0002265)
1: (-0.0001203, 0.0014702, -0.0001203, 0.0014702, -0.0008239, 0.0008239)
2: (0.0141382, 0.0165201, 0.0141382, 0.0165201, -0.0012175, 0.0012175)
3: (0.0000044, 0.0017955, 0.0000044, 0.0017955, -0.0009094, 0.0009094)
4: (-0.0043755, -0.0027234, -0.0043755, -0.0027234, -0.0009049, 0.0009049)
5: (0.0079426, 0.0097304, 0.0079426, 0.0097304, -0.0009072, 0.0009072)
6: (0.0092609, 0.0099356, 0.0092609, 0.0099356, -0.0005490, 0.0005490)
7: (-0.0195231, -0.0156420, -0.0195231, -0.0156420, -0.0019098, 0.0019098)
8: (0.9678549, 0.9789748, 0.9678549, 0.9789748, -0.0056980, 0.0056980)
9: (0.0038325, 0.0071006, 0.0038325, 0.0071006, -0.0016261, 0.0016261)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 1.32 = 2.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0045405, upper bound: 0.0045405

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044760, upper bound: 0.0044407
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044407, upper bound: 0.0044760
time: 0.49 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.08 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 8, lower bound: -0.0044760, upper bound: 0.0044407
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 8, lower bound: -0.0044407, upper bound: 0.0044760

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0003077, 0.0000969, -0.0003077, 0.0000969, -0.0002215, 0.0002216
1: -0.0001203, 0.0014702, -0.0001203, 0.0014702, -0.0007941, 0.0007949
2: 0.0141382, 0.0165201, 0.0141382, 0.0165201, -0.0011739, 0.0011726
3: 0.0000044, 0.0017955, 0.0000044, 0.0017955, -0.0008769, 0.0008759
4: -0.0043755, -0.0027234, -0.0043755, -0.0027234, -0.0008737, 0.0008728
5: 0.0079426, 0.0097304, 0.0079426, 0.0097304, -0.0008747, 0.0008738
6: 0.0092609, 0.0099356, 0.0092609, 0.0099356, -0.0005371, 0.0005374
7: -0.0195231, -0.0156420, -0.0195231, -0.0156420, -0.0018374, 0.0018394
8: 0.9678549, 0.9789748, 0.9678549, 0.9789748, -0.0054891, 0.0054950
9: 0.0038325, 0.0071006, 0.0038325, 0.0071006, -0.0015672, 0.0015655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0038701, upper bound: 0.0038481
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0038701, upper bound: 0.0038481
time: 0.45 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0003077, 0.0000969, -0.0003077, 0.0000969, -0.0002216, 0.0002215
1: -0.0001203, 0.0014702, -0.0001203, 0.0014702, -0.0007949, 0.0007941
2: 0.0141382, 0.0165201, 0.0141382, 0.0165201, -0.0011726, 0.0011739
3: 0.0000044, 0.0017955, 0.0000044, 0.0017955, -0.0008759, 0.0008769
4: -0.0043755, -0.0027234, -0.0043755, -0.0027234, -0.0008728, 0.0008737
5: 0.0079426, 0.0097304, 0.0079426, 0.0097304, -0.0008738, 0.0008747
6: 0.0092609, 0.0099356, 0.0092609, 0.0099356, -0.0005374, 0.0005371
7: -0.0195231, -0.0156420, -0.0195231, -0.0156420, -0.0018394, 0.0018374
8: 0.9678549, 0.9789748, 0.9678549, 0.9789748, -0.0054950, 0.0054891
9: 0.0038325, 0.0071006, 0.0038325, 0.0071006, -0.0015655, 0.0015672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0038481, upper bound: 0.0038701
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0038481, upper bound: 0.0038701
time: 0.50 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 8, lower bound: -0.0038701, upper bound: 0.0038481
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 8, lower bound: -0.0038701, upper bound: 0.0038481
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 8, lower bound: -0.0038481, upper bound: 0.0038701
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 8, lower bound: -0.0038481, upper bound: 0.0038701

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0003077, 0.0000969, -0.0003077, 0.0000969, -0.0002205, 0.0002213
1: -0.0001203, 0.0014702, -0.0001203, 0.0014702, -0.0007893, 0.0007930
2: 0.0141382, 0.0165201, 0.0141382, 0.0165201, -0.0011710, 0.0011654
3: 0.0000044, 0.0017955, 0.0000044, 0.0017955, -0.0008747, 0.0008705
4: -0.0043755, -0.0027234, -0.0043755, -0.0027234, -0.0008717, 0.0008678
5: 0.0079426, 0.0097304, 0.0079426, 0.0097304, -0.0008726, 0.0008684
6: 0.0092609, 0.0099356, 0.0092609, 0.0099356, -0.0005350, 0.0005366
7: -0.0195231, -0.0156420, -0.0195231, -0.0156420, -0.0018256, 0.0018347
8: 0.9678549, 0.9789748, 0.9678549, 0.9789748, -0.0054553, 0.0054816
9: 0.0038325, 0.0071006, 0.0038325, 0.0071006, -0.0015633, 0.0015555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026191, upper bound: 0.0026713
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026828, upper bound: 0.0026100
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0003077, 0.0000969, -0.0003077, 0.0000969, -0.0002212, 0.0002216
1: -0.0001203, 0.0014702, -0.0001203, 0.0014702, -0.0007922, 0.0007949
2: 0.0141382, 0.0165201, 0.0141382, 0.0165201, -0.0011739, 0.0011698
3: 0.0000044, 0.0017955, 0.0000044, 0.0017955, -0.0008769, 0.0008738
4: -0.0043755, -0.0027234, -0.0043755, -0.0027234, -0.0008737, 0.0008708
5: 0.0079426, 0.0097304, 0.0079426, 0.0097304, -0.0008747, 0.0008716
6: 0.0092609, 0.0099356, 0.0092609, 0.0099356, -0.0005363, 0.0005374
7: -0.0195231, -0.0156420, -0.0195231, -0.0156420, -0.0018327, 0.0018394
8: 0.9678549, 0.9789748, 0.9678549, 0.9789748, -0.0054757, 0.0054950
9: 0.0038325, 0.0071006, 0.0038325, 0.0071006, -0.0015672, 0.0015615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026191, upper bound: 0.0026713
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026828, upper bound: 0.0026100
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0003077, 0.0000969, -0.0003077, 0.0000969, -0.0002204, 0.0002212
1: -0.0001203, 0.0014702, -0.0001203, 0.0014702, -0.0007886, 0.0007922
2: 0.0141382, 0.0165201, 0.0141382, 0.0165201, -0.0011698, 0.0011644
3: 0.0000044, 0.0017955, 0.0000044, 0.0017955, -0.0008738, 0.0008698
4: -0.0043755, -0.0027234, -0.0043755, -0.0027234, -0.0008708, 0.0008671
5: 0.0079426, 0.0097304, 0.0079426, 0.0097304, -0.0008716, 0.0008676
6: 0.0092609, 0.0099356, 0.0092609, 0.0099356, -0.0005347, 0.0005363
7: -0.0195231, -0.0156420, -0.0195231, -0.0156420, -0.0018240, 0.0018327
8: 0.9678549, 0.9789748, 0.9678549, 0.9789748, -0.0054508, 0.0054757
9: 0.0038325, 0.0071006, 0.0038325, 0.0071006, -0.0015615, 0.0015542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026100, upper bound: 0.0026828
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026713, upper bound: 0.0026191
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0003077, 0.0000969, -0.0003077, 0.0000969, -0.0002213, 0.0002215
1: -0.0001203, 0.0014702, -0.0001203, 0.0014702, -0.0007930, 0.0007941
2: 0.0141382, 0.0165201, 0.0141382, 0.0165201, -0.0011726, 0.0011710
3: 0.0000044, 0.0017955, 0.0000044, 0.0017955, -0.0008759, 0.0008747
4: -0.0043755, -0.0027234, -0.0043755, -0.0027234, -0.0008728, 0.0008717
5: 0.0079426, 0.0097304, 0.0079426, 0.0097304, -0.0008738, 0.0008726
6: 0.0092609, 0.0099356, 0.0092609, 0.0099356, -0.0005366, 0.0005371
7: -0.0195231, -0.0156420, -0.0195231, -0.0156420, -0.0018347, 0.0018374
8: 0.9678549, 0.9789748, 0.9678549, 0.9789748, -0.0054816, 0.0054891
9: 0.0038325, 0.0071006, 0.0038325, 0.0071006, -0.0015655, 0.0015633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026100, upper bound: 0.0026828
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026713, upper bound: 0.0026191
time: 0.43 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 8, lower bound: -0.0026191, upper bound: 0.0026713
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 8, lower bound: -0.0026828, upper bound: 0.0026100
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 8, lower bound: -0.0026191, upper bound: 0.0026713
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 8, lower bound: -0.0026828, upper bound: 0.0026100
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 8, lower bound: -0.0026100, upper bound: 0.0026828
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 8, lower bound: -0.0026713, upper bound: 0.0026191
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 8, lower bound: -0.0026100, upper bound: 0.0026828
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 8, lower bound: -0.0026713, upper bound: 0.0026191

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.75 + 15.23 = 17.97 seconds
