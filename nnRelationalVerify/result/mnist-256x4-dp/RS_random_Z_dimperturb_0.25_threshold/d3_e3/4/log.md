## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00085666


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020829, 0.0020829)
1: (-0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005190, 0.0005190)
2: (0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027504, 0.0027504)
3: (-0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012519, 0.0012519)
4: (0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005323, 0.0005323)
5: (0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034593, 0.0034593)
6: (0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008780, 0.0008780)
7: (-0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022717, 0.0022717)
8: (-0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011947, 0.0011947)
9: (-0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013853, 0.0013853)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.65 + 1.49 = 3.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0012150, upper bound: 0.0012151

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012091, upper bound: 0.0012002
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012003, upper bound: 0.0012091
time: 0.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.23
Output dim: 0, lower bound: -0.0012091, upper bound: 0.0012002
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.23
Output dim: 0, lower bound: -0.0012003, upper bound: 0.0012091

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020826, 0.0020816
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005189, 0.0005187
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027488, 0.0027500
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012517, 0.0012511
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005320, 0.0005323
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034573, 0.0034588
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008779, 0.0008775
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022713, 0.0022703
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011945, 0.0011939
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013844, 0.0013850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011899, upper bound: 0.0011739
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011820, upper bound: 0.0011813
time: 0.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020816, 0.0020826
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005187, 0.0005189
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027500, 0.0027488
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012511, 0.0012517
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005323, 0.0005320
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034588, 0.0034573
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008775, 0.0008779
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022703, 0.0022713
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011939, 0.0011945
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013850, 0.0013844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0009006
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0009006
time: 0.54 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0011899, upper bound: 0.0011739
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0011820, upper bound: 0.0011813
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0009006
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0009004, upper bound: 0.0009006

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020830, 0.0020812
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005190, 0.0005186
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027483, 0.0027506
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012520, 0.0012509
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005319, 0.0005324
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034566, 0.0034595
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008781, 0.0008773
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022718, 0.0022699
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011947, 0.0011937
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013842, 0.0013854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008819, upper bound: 0.0008754
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008819, upper bound: 0.0008754
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020820, 0.0020821
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005188, 0.0005188
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027494, 0.0027493
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012514, 0.0012514
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005321, 0.0005321
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034580, 0.0034579
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008776, 0.0008777
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022707, 0.0022708
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011942, 0.0011942
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013847, 0.0013847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011698, upper bound: 0.0011534
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011554, upper bound: 0.0011691
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020775, 0.0020983
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005177, 0.0005228
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027708, 0.0027433
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012486, 0.0012611
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005363, 0.0005310
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034849, 0.0034504
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008757, 0.0008845
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022658, 0.0022885
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011916, 0.0012035
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013955, 0.0013817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008671, upper bound: 0.0008638
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008631, upper bound: 0.0008674
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020816, 0.0020784
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005187, 0.0005179
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027445, 0.0027488
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012511, 0.0012492
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005312, 0.0005320
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034519, 0.0034573
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008775, 0.0008761
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022703, 0.0022668
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011939, 0.0011921
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013823, 0.0013844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008901, upper bound: 0.0008758
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008758, upper bound: 0.0008903
time: 0.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0008819, upper bound: 0.0008754
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0008819, upper bound: 0.0008754
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0011698, upper bound: 0.0011534
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0011554, upper bound: 0.0011691
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0008671, upper bound: 0.0008638
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0008631, upper bound: 0.0008674
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0008901, upper bound: 0.0008758
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0008758, upper bound: 0.0008903

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020789, 0.0020959
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005180, 0.0005222
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027676, 0.0027451
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012495, 0.0012597
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005357, 0.0005313
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034809, 0.0034526
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008763, 0.0008835
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022673, 0.0022859
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011923, 0.0012021
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013939, 0.0013826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007865, upper bound: 0.0007144
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007154, upper bound: 0.0007793
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020830, 0.0020771
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005190, 0.0005176
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027428, 0.0027506
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012520, 0.0012484
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005309, 0.0005324
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034497, 0.0034595
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008781, 0.0008756
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022718, 0.0022654
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011947, 0.0011913
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013814, 0.0013854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008360, upper bound: 0.0008129
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008155, upper bound: 0.0008301
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020646, 0.0020530
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005144, 0.0005116
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027110, 0.0027262
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012409, 0.0012339
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005247, 0.0005277
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034097, 0.0034289
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008703, 0.0008654
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022517, 0.0022391
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011842, 0.0011775
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013654, 0.0013731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008647, upper bound: 0.0008573
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008647, upper bound: 0.0008573
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020529, 0.0020647
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005115, 0.0005145
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027264, 0.0027109
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012339, 0.0012409
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005277, 0.0005247
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034291, 0.0034095
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008654, 0.0008703
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022390, 0.0022518
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011775, 0.0011842
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013731, 0.0013653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008540, upper bound: 0.0008715
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008540, upper bound: 0.0008715
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020456, 0.0020588
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005097, 0.0005130
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027187, 0.0027012
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012294, 0.0012374
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005262, 0.0005228
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034194, 0.0033973
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008623, 0.0008679
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022310, 0.0022455
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011733, 0.0011809
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013693, 0.0013604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008568, upper bound: 0.0008398
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008431, upper bound: 0.0008533
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020381, 0.0020664
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005079, 0.0005149
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027286, 0.0026913
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012250, 0.0012419
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005281, 0.0005209
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034319, 0.0033850
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008592, 0.0008711
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022229, 0.0022537
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011690, 0.0011852
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013743, 0.0013555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008443, upper bound: 0.0008413
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008375, upper bound: 0.0008486
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020646, 0.0020497
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005145, 0.0005107
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027067, 0.0027263
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012409, 0.0012320
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005239, 0.0005277
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034043, 0.0034290
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008703, 0.0008640
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022518, 0.0022355
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011842, 0.0011756
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013632, 0.0013731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007946, upper bound: 0.0007119
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007227, upper bound: 0.0007792
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020530, 0.0020614
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005115, 0.0005137
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027221, 0.0027109
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012339, 0.0012390
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005269, 0.0005247
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034237, 0.0034096
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008654, 0.0008690
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022391, 0.0022483
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011775, 0.0011823
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013710, 0.0013654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008298, upper bound: 0.0008235
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008104, upper bound: 0.0008443
time: 0.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0007865, upper bound: 0.0007144
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0007154, upper bound: 0.0007793
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0008360, upper bound: 0.0008129
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0008155, upper bound: 0.0008301
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0008647, upper bound: 0.0008573
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0008647, upper bound: 0.0008573
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0008540, upper bound: 0.0008715
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0008540, upper bound: 0.0008715
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0008568, upper bound: 0.0008398
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0008431, upper bound: 0.0008533
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0008443, upper bound: 0.0008413
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0008375, upper bound: 0.0008486
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0007946, upper bound: 0.0007119
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0007227, upper bound: 0.0007792
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0008298, upper bound: 0.0008235
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0008104, upper bound: 0.0008443

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020603, 0.0020686
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005134, 0.0005154
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027315, 0.0027206
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012383, 0.0012433
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005287, 0.0005266
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034356, 0.0034218
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008685, 0.0008720
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022471, 0.0022561
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011817, 0.0011865
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013758, 0.0013702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008193, upper bound: 0.0007921
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008025, upper bound: 0.0008113
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020646, 0.0020487
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005144, 0.0005105
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027053, 0.0027262
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012409, 0.0012313
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005236, 0.0005277
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034026, 0.0034289
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008703, 0.0008636
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022517, 0.0022344
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011842, 0.0011751
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013625, 0.0013731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008309, upper bound: 0.0008213
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008272, upper bound: 0.0008246
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020487, 0.0020801
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005105, 0.0005183
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027467, 0.0027052
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012313, 0.0012502
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005316, 0.0005236
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034546, 0.0034025
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008636, 0.0008768
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022343, 0.0022686
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011750, 0.0011930
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013834, 0.0013625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007561, upper bound: 0.0007044
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006930, upper bound: 0.0007760
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020529, 0.0020604
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005115, 0.0005134
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0027207, 0.0027109
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012339, 0.0012384
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005266, 0.0005247
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0034220, 0.0034095
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008654, 0.0008685
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022390, 0.0022472
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011775, 0.0011818
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013703, 0.0013653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008080, upper bound: 0.0008051
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007920, upper bound: 0.0008254
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9938012, 0.9964839, 0.9938012, 0.9964839, -0.0020239, 0.0020255
1: -0.0028085, -0.0021401, -0.0028085, -0.0021401, -0.0005043, 0.0005047
2: 0.0012872, 0.0048297, 0.0012872, 0.0048297, -0.0026746, 0.0026725
3: -0.0034714, -0.0018590, -0.0034714, -0.0018590, -0.0012164, 0.0012174
4: 0.0007770, 0.0014627, 0.0007770, 0.0014627, -0.0005177, 0.0005173
5: 0.0005785, 0.0050340, 0.0005785, 0.0050340, -0.0033640, 0.0033613
6: 0.0002632, 0.0013940, 0.0002632, 0.0013940, -0.0008531, 0.0008538
7: -0.0024568, 0.0004691, -0.0024568, 0.0004691, -0.0022073, 0.0022091
8: -0.0008561, 0.0006826, -0.0008561, 0.0006826, -0.0011608, 0.0011617
9: -0.0026553, -0.0008711, -0.0026553, -0.0008711, -0.0013471, 0.0013460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008381, upper bound: 0.0008192
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008310, upper bound: 0.0008214
time: 0.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.70 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 0, lower bound: -0.0008193, upper bound: 0.0007921
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 0, lower bound: -0.0008025, upper bound: 0.0008113
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 0, lower bound: -0.0008309, upper bound: 0.0008213
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 0, lower bound: -0.0008272, upper bound: 0.0008246
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 0, lower bound: -0.0007561, upper bound: 0.0007044
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 0, lower bound: -0.0006930, upper bound: 0.0007760
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 0, lower bound: -0.0008080, upper bound: 0.0008051
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 0, lower bound: -0.0007920, upper bound: 0.0008254
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 0, lower bound: -0.0008381, upper bound: 0.0008192
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 0, lower bound: -0.0008310, upper bound: 0.0008214

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.14 + 56.33 = 59.47 seconds
