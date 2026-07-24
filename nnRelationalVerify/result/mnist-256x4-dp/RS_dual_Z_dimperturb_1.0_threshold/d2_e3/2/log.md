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
Threshold: 0.0157495625


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293)
1: (-0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382)
2: (0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210)
3: (-0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023)
4: (0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575)
5: (0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635)
6: (-0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247)
7: (-0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719)
8: (-0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123)
9: (-0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 2.24 = 3.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0179995, upper bound: 0.0179995

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0175203, upper bound: 0.0175074
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0175074, upper bound: 0.0175203
time: 1.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.73
Output dim: 0, lower bound: -0.0175203, upper bound: 0.0175074
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.73
Output dim: 0, lower bound: -0.0175074, upper bound: 0.0175203

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0173869, upper bound: 0.0173748
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0173869, upper bound: 0.0173748
time: 1.44 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0173748, upper bound: 0.0173869
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0173748, upper bound: 0.0173869
time: 1.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.70
Output dim: 0, lower bound: -0.0173869, upper bound: 0.0173748
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.70
Output dim: 0, lower bound: -0.0173869, upper bound: 0.0173748
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.70
Output dim: 0, lower bound: -0.0173748, upper bound: 0.0173869
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.70
Output dim: 0, lower bound: -0.0173748, upper bound: 0.0173869

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171340, upper bound: 0.0171468
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171544, upper bound: 0.0171287
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171340, upper bound: 0.0171468
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171544, upper bound: 0.0171287
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171287, upper bound: 0.0171544
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171468, upper bound: 0.0171340
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171287, upper bound: 0.0171544
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171468, upper bound: 0.0171340
time: 1.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.54 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -0.0171340, upper bound: 0.0171468
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -0.0171544, upper bound: 0.0171287
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -0.0171340, upper bound: 0.0171468
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -0.0171544, upper bound: 0.0171287
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -0.0171287, upper bound: 0.0171544
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -0.0171468, upper bound: 0.0171340
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -0.0171287, upper bound: 0.0171544
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -0.0171468, upper bound: 0.0171340

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170507, upper bound: 0.0170681
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170526, upper bound: 0.0170636
time: 1.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170669, upper bound: 0.0170483
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170717, upper bound: 0.0170470
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170487, upper bound: 0.0170686
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170517, upper bound: 0.0170653
time: 1.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170655, upper bound: 0.0170500
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170709, upper bound: 0.0170489
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170489, upper bound: 0.0170709
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170500, upper bound: 0.0170655
time: 2.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170653, upper bound: 0.0170517
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170687, upper bound: 0.0170487
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170470, upper bound: 0.0170717
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170483, upper bound: 0.0170669
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170636, upper bound: 0.0170526
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170681, upper bound: 0.0170507
time: 2.08 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170507, upper bound: 0.0170681
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170526, upper bound: 0.0170636
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170669, upper bound: 0.0170483
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170717, upper bound: 0.0170470
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170487, upper bound: 0.0170686
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170517, upper bound: 0.0170653
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170655, upper bound: 0.0170500
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170709, upper bound: 0.0170489
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170489, upper bound: 0.0170709
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170500, upper bound: 0.0170655
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170653, upper bound: 0.0170517
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170687, upper bound: 0.0170487
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170470, upper bound: 0.0170717
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170483, upper bound: 0.0170669
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170636, upper bound: 0.0170526
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.00
Output dim: 0, lower bound: -0.0170681, upper bound: 0.0170507

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167585, upper bound: 0.0168950
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168721, upper bound: 0.0167536
time: 3.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167585, upper bound: 0.0168883
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168753, upper bound: 0.0167536
time: 2.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167647, upper bound: 0.0168709
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168907, upper bound: 0.0167470
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167647, upper bound: 0.0168686
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168988, upper bound: 0.0167470
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167485, upper bound: 0.0168958
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168704, upper bound: 0.0167616
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167485, upper bound: 0.0168890
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168747, upper bound: 0.0167616
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167547, upper bound: 0.0168720
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168899, upper bound: 0.0167553
time: 1.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167549, upper bound: 0.0168704
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168983, upper bound: 0.0167553
time: 1.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167553, upper bound: 0.0168983
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168704, upper bound: 0.0167549
time: 1.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167553, upper bound: 0.0168899
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168720, upper bound: 0.0167547
time: 2.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167616, upper bound: 0.0168747
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168890, upper bound: 0.0167485
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167616, upper bound: 0.0168704
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168958, upper bound: 0.0167485
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167470, upper bound: 0.0168988
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168686, upper bound: 0.0167647
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167470, upper bound: 0.0168907
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168709, upper bound: 0.0167647
time: 1.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167536, upper bound: 0.0168753
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168883, upper bound: 0.0167585
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167536, upper bound: 0.0168721
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168950, upper bound: 0.0167585
time: 1.88 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167585, upper bound: 0.0168950
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168721, upper bound: 0.0167536
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167585, upper bound: 0.0168883
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168753, upper bound: 0.0167536
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167647, upper bound: 0.0168709
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168907, upper bound: 0.0167470
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167647, upper bound: 0.0168686
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168988, upper bound: 0.0167470
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167485, upper bound: 0.0168958
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168704, upper bound: 0.0167616
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167485, upper bound: 0.0168890
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168747, upper bound: 0.0167616
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167547, upper bound: 0.0168720
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168899, upper bound: 0.0167553
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167549, upper bound: 0.0168704
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168983, upper bound: 0.0167553
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167553, upper bound: 0.0168983
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168704, upper bound: 0.0167549
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167553, upper bound: 0.0168899
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168720, upper bound: 0.0167547
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167616, upper bound: 0.0168747
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168890, upper bound: 0.0167485
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167616, upper bound: 0.0168704
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168958, upper bound: 0.0167485
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167470, upper bound: 0.0168988
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168686, upper bound: 0.0167647
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167470, upper bound: 0.0168907
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168709, upper bound: 0.0167647
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167536, upper bound: 0.0168753
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168883, upper bound: 0.0167585
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0167536, upper bound: 0.0168721
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -0.0168950, upper bound: 0.0167585

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
time: 2.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
time: 1.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
time: 1.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145371, upper bound: 0.0146793
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145371, upper bound: 0.0146793
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146792
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145371, upper bound: 0.0146793
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
time: 1.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146793
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145371, upper bound: 0.0146793
time: 2.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145371, upper bound: 0.0146793
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145371, upper bound: 0.0146793
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
time: 1.89 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145371, upper bound: 0.0146793
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145371, upper bound: 0.0146793
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146792
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145371, upper bound: 0.0146793
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145773, upper bound: 0.0146482
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146792, upper bound: 0.0145371
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145775, upper bound: 0.0146464
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146806, upper bound: 0.0145370
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146806
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146464, upper bound: 0.0145775
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145370, upper bound: 0.0146793
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145371, upper bound: 0.0146793
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145371, upper bound: 0.0146793
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0145371, upper bound: 0.0146793
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 0, lower bound: -0.0146482, upper bound: 0.0145773

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.54 + 302.99 = 306.53 seconds
