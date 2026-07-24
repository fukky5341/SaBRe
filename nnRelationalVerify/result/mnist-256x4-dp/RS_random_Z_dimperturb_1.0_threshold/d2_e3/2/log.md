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
execution time: IAR + RelationalAnalysis = 1.29 + 2.23 = 3.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0179995, upper bound: 0.0179995

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0176846, upper bound: 0.0176763
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0176763, upper bound: 0.0176845
time: 1.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.80 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.80
Output dim: 0, lower bound: -0.0176846, upper bound: 0.0176763
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.80
Output dim: 0, lower bound: -0.0176763, upper bound: 0.0176845

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0174651, upper bound: 0.0174208
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0174328, upper bound: 0.0174601
time: 1.38 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0175774, upper bound: 0.0176152
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0176062, upper bound: 0.0175933
time: 1.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 0, lower bound: -0.0174651, upper bound: 0.0174208
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 0, lower bound: -0.0174328, upper bound: 0.0174601
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 0, lower bound: -0.0175774, upper bound: 0.0176152
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 0, lower bound: -0.0176062, upper bound: 0.0175933

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
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 231

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0173130, upper bound: 0.0172728
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0173130, upper bound: 0.0172737
time: 1.69 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0136890, upper bound: 0.0137073
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0136890, upper bound: 0.0137073
time: 1.04 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0173370, upper bound: 0.0173863
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0173507, upper bound: 0.0173708
time: 1.46 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 231

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164413, upper bound: 0.0164252
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164413, upper bound: 0.0164252
time: 1.53 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.27 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0173130, upper bound: 0.0172728
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0173130, upper bound: 0.0172737
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0136890, upper bound: 0.0137073
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0136890, upper bound: 0.0137073
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0173370, upper bound: 0.0173863
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0173507, upper bound: 0.0173708
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0164413, upper bound: 0.0164252
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0164413, upper bound: 0.0164252

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146346, upper bound: 0.0146028
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146346, upper bound: 0.0146028
time: 1.08 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0169793, upper bound: 0.0170999
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171375, upper bound: 0.0169140
time: 1.42 seconds

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
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0172958, upper bound: 0.0172800
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0172336, upper bound: 0.0173450
time: 1.40 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0172077, upper bound: 0.0172155
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0172070, upper bound: 0.0172176
time: 1.52 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162006, upper bound: 0.0162053
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162213, upper bound: 0.0161927
time: 1.52 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163130, upper bound: 0.0162900
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163130, upper bound: 0.0162909
time: 1.95 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.09
Output dim: 0, lower bound: -0.0146346, upper bound: 0.0146028
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.09
Output dim: 0, lower bound: -0.0146346, upper bound: 0.0146028
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 0, lower bound: -0.0169793, upper bound: 0.0170999
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 0, lower bound: -0.0171375, upper bound: 0.0169140
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 0, lower bound: -0.0172958, upper bound: 0.0172800
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 0, lower bound: -0.0172336, upper bound: 0.0173450
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 0, lower bound: -0.0172077, upper bound: 0.0172155
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 0, lower bound: -0.0172070, upper bound: 0.0172176
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 0, lower bound: -0.0162006, upper bound: 0.0162053
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 0, lower bound: -0.0162213, upper bound: 0.0161927
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 0, lower bound: -0.0163130, upper bound: 0.0162900
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 0, lower bound: -0.0163130, upper bound: 0.0162909

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0169550, upper bound: 0.0170767
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0169562, upper bound: 0.0170767
time: 1.47 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0133790, upper bound: 0.0132585
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0133790, upper bound: 0.0132585
time: 1.22 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 231

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0172524, upper bound: 0.0170880
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0170722, upper bound: 0.0172369
time: 1.76 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159157, upper bound: 0.0160235
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159157, upper bound: 0.0160235
time: 1.52 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168677, upper bound: 0.0168226
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168166, upper bound: 0.0168857
time: 1.49 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171835, upper bound: 0.0171941
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171845, upper bound: 0.0171941
time: 1.53 seconds

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160750, upper bound: 0.0160742
time: 2.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160750, upper bound: 0.0160746
time: 2.63 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 231

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159000, upper bound: 0.0158254
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158475, upper bound: 0.0158739
time: 1.49 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 231

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162765, upper bound: 0.0160753
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0161068, upper bound: 0.0162540
time: 1.95 seconds

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162713, upper bound: 0.0161844
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162080, upper bound: 0.0162502
time: 1.82 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.37 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0169550, upper bound: 0.0170767
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0169562, upper bound: 0.0170767
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0133790, upper bound: 0.0132585
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0133790, upper bound: 0.0132585
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0172524, upper bound: 0.0170880
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0170722, upper bound: 0.0172369
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0159157, upper bound: 0.0160235
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0159157, upper bound: 0.0160235
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0168677, upper bound: 0.0168226
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0168166, upper bound: 0.0168857
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0171835, upper bound: 0.0171941
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0171845, upper bound: 0.0171941
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0160750, upper bound: 0.0160742
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0160750, upper bound: 0.0160746
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0159000, upper bound: 0.0158254
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0158475, upper bound: 0.0158739
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0162765, upper bound: 0.0160753
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0161068, upper bound: 0.0162540
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0162713, upper bound: 0.0161844
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 0, lower bound: -0.0162080, upper bound: 0.0162502

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0142278, upper bound: 0.0142017
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0142278, upper bound: 0.0142017
time: 1.23 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0132566, upper bound: 0.0133382
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0132566, upper bound: 0.0133382
time: 1.22 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168622, upper bound: 0.0168220
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168148, upper bound: 0.0168805
time: 1.71 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167865, upper bound: 0.0170586
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168856, upper bound: 0.0169278
time: 1.49 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0156309, upper bound: 0.0158154
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0157024, upper bound: 0.0156869
time: 1.80 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0130495, upper bound: 0.0130740
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0130495, upper bound: 0.0130740
time: 1.52 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168276, upper bound: 0.0167169
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167588, upper bound: 0.0167813
time: 1.55 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155304, upper bound: 0.0155820
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155304, upper bound: 0.0155820
time: 1.50 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 231

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158977, upper bound: 0.0159204
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158977, upper bound: 0.0159204
time: 1.52 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0169659, upper bound: 0.0169457
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0169342, upper bound: 0.0169694
time: 1.74 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0143440, upper bound: 0.0143326
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0143440, upper bound: 0.0143326
time: 1.59 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158564, upper bound: 0.0158284
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158251, upper bound: 0.0158499
time: 1.54 seconds

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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129863, upper bound: 0.0129854
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129863, upper bound: 0.0129854
time: 1.07 seconds

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 231

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155393, upper bound: 0.0156612
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0156364, upper bound: 0.0155745
time: 1.91 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162348, upper bound: 0.0159932
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0161709, upper bound: 0.0160320
time: 1.96 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160755, upper bound: 0.0162283
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160829, upper bound: 0.0162294
time: 1.50 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159314, upper bound: 0.0159801
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160679, upper bound: 0.0158867
time: 1.51 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144960, upper bound: 0.0145385
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144960, upper bound: 0.0145385
time: 1.69 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.59 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0142278, upper bound: 0.0142017
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0142278, upper bound: 0.0142017
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0132566, upper bound: 0.0133382
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0132566, upper bound: 0.0133382
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0168622, upper bound: 0.0168220
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0168148, upper bound: 0.0168805
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0167865, upper bound: 0.0170586
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0168856, upper bound: 0.0169278
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0156309, upper bound: 0.0158154
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0157024, upper bound: 0.0156869
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0130495, upper bound: 0.0130740
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0130495, upper bound: 0.0130740
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0168276, upper bound: 0.0167169
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0167588, upper bound: 0.0167813
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0155304, upper bound: 0.0155820
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0155304, upper bound: 0.0155820
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0158977, upper bound: 0.0159204
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0158977, upper bound: 0.0159204
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0169659, upper bound: 0.0169457
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0169342, upper bound: 0.0169694
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0143440, upper bound: 0.0143326
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0143440, upper bound: 0.0143326
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0158564, upper bound: 0.0158284
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0158251, upper bound: 0.0158499
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0129863, upper bound: 0.0129854
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0129863, upper bound: 0.0129854
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0155393, upper bound: 0.0156612
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0156364, upper bound: 0.0155745
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0162348, upper bound: 0.0159932
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0161709, upper bound: 0.0160320
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0160755, upper bound: 0.0162283
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0160829, upper bound: 0.0162294
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0159314, upper bound: 0.0159801
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0160679, upper bound: 0.0158867
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0144960, upper bound: 0.0145385
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -0.0144960, upper bound: 0.0145385

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167233, upper bound: 0.0164567
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166866, upper bound: 0.0164964
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0169162, upper bound: 0.0167349
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168093, upper bound: 0.0167924
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0148044, upper bound: 0.0150116
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0148044, upper bound: 0.0150116
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0157215, upper bound: 0.0157719
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0157215, upper bound: 0.0157719
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0140568, upper bound: 0.0141056
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0140568, upper bound: 0.0141056
time: 1.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167482, upper bound: 0.0165761
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165876, upper bound: 0.0166336
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164769, upper bound: 0.0165957
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165691, upper bound: 0.0164752
time: 2.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129872, upper bound: 0.0129942
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129872, upper bound: 0.0129942
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155436, upper bound: 0.0155141
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155035, upper bound: 0.0155633
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0156791, upper bound: 0.0156738
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0156791, upper bound: 0.0156738
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168247, upper bound: 0.0168685
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168203, upper bound: 0.0168688
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158324, upper bound: 0.0158035
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0157998, upper bound: 0.0158047
time: 1.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 231

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0140974, upper bound: 0.0141076
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0140974, upper bound: 0.0141076
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155546, upper bound: 0.0153330
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155546, upper bound: 0.0153331
time: 2.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158405, upper bound: 0.0156570
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0157977, upper bound: 0.0157016
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0157667, upper bound: 0.0160243
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158729, upper bound: 0.0159114
time: 2.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0154015, upper bound: 0.0155706
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0154012, upper bound: 0.0155706
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0152665, upper bound: 0.0152874
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0152665, upper bound: 0.0152874
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159568, upper bound: 0.0157757
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159547, upper bound: 0.0157766
time: 1.63 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.69 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0167233, upper bound: 0.0164567
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0166866, upper bound: 0.0164964
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0169162, upper bound: 0.0167349
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0168093, upper bound: 0.0167924
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0148044, upper bound: 0.0150116
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0148044, upper bound: 0.0150116
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0157215, upper bound: 0.0157719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0157215, upper bound: 0.0157719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0140568, upper bound: 0.0141056
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0140568, upper bound: 0.0141056
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0167482, upper bound: 0.0165761
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0165876, upper bound: 0.0166336
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0164769, upper bound: 0.0165957
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0165691, upper bound: 0.0164752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0129872, upper bound: 0.0129942
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0129872, upper bound: 0.0129942
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0155436, upper bound: 0.0155141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0155035, upper bound: 0.0155633
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0156791, upper bound: 0.0156738
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0156791, upper bound: 0.0156738
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0168247, upper bound: 0.0168685
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0168203, upper bound: 0.0168688
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0158324, upper bound: 0.0158035
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0157998, upper bound: 0.0158047
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0140974, upper bound: 0.0141076
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0140974, upper bound: 0.0141076
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0155546, upper bound: 0.0153330
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0155546, upper bound: 0.0153331
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0158405, upper bound: 0.0156570
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0157977, upper bound: 0.0157016
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0157667, upper bound: 0.0160243
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0158729, upper bound: 0.0159114
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0154015, upper bound: 0.0155706
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0154012, upper bound: 0.0155706
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0152665, upper bound: 0.0152874
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0152665, upper bound: 0.0152874
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0159568, upper bound: 0.0157757
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.69
Output dim: 0, lower bound: -0.0159547, upper bound: 0.0157766

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0060937
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153792, upper bound: 0.0151229
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153792, upper bound: 0.0151229
time: 2.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165139, upper bound: 0.0163249
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165136, upper bound: 0.0163249
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167722, upper bound: 0.0165839
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167697, upper bound: 0.0165839
time: 2.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0139784, upper bound: 0.0139700
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0139784, upper bound: 0.0139700
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0150427, upper bound: 0.0150732
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0150425, upper bound: 0.0150732
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0156225, upper bound: 0.0157480
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0156979, upper bound: 0.0157477
time: 1.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0154632, upper bound: 0.0153233
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0154632, upper bound: 0.0153233
time: 1.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0147208, upper bound: 0.0147219
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0147208, upper bound: 0.0147219
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153680, upper bound: 0.0154633
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153680, upper bound: 0.0154633
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0060975
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164790, upper bound: 0.0163690
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164751, upper bound: 0.0163742
time: 1.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165035, upper bound: 0.0166979
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166513, upper bound: 0.0165568
time: 1.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163561, upper bound: 0.0163813
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163538, upper bound: 0.0163846
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0130070, upper bound: 0.0130114
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0130070, upper bound: 0.0130114
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0156744, upper bound: 0.0156131
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0156744, upper bound: 0.0157267
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0154736, upper bound: 0.0154577
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0156354, upper bound: 0.0154472
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0131708, upper bound: 0.0130916
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0131708, upper bound: 0.0130916
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0155575, upper bound: 0.0158159
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0155627, upper bound: 0.0157968
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0157633, upper bound: 0.0158001
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0157613, upper bound: 0.0158006
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 231

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159200, upper bound: 0.0156033
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0157523, upper bound: 0.0157383
time: 1.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0157311, upper bound: 0.0155160
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0157048, upper bound: 0.0155646
time: 1.85 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 6.82 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0153792, upper bound: 0.0151229
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0153792, upper bound: 0.0151229
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0165139, upper bound: 0.0163249
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0165136, upper bound: 0.0163249
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0167722, upper bound: 0.0165839
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0167697, upper bound: 0.0165839
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0139784, upper bound: 0.0139700
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0139784, upper bound: 0.0139700
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0150427, upper bound: 0.0150732
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0150425, upper bound: 0.0150732
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0156225, upper bound: 0.0157480
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0156979, upper bound: 0.0157477
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0154632, upper bound: 0.0153233
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0154632, upper bound: 0.0153233
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0147208, upper bound: 0.0147219
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0147208, upper bound: 0.0147219
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0153680, upper bound: 0.0154633
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0153680, upper bound: 0.0154633
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0164790, upper bound: 0.0163690
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0164751, upper bound: 0.0163742
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0165035, upper bound: 0.0166979
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0166513, upper bound: 0.0165568
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0163561, upper bound: 0.0163813
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0163538, upper bound: 0.0163846
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0130070, upper bound: 0.0130114
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0130070, upper bound: 0.0130114
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0156744, upper bound: 0.0156131
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0156744, upper bound: 0.0157267
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0154736, upper bound: 0.0154577
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0156354, upper bound: 0.0154472
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0131708, upper bound: 0.0130916
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0131708, upper bound: 0.0130916
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0155575, upper bound: 0.0158159
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0155627, upper bound: 0.0157968
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0157633, upper bound: 0.0158001
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0157613, upper bound: 0.0158006
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0159200, upper bound: 0.0156033
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0157523, upper bound: 0.0157383
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0157311, upper bound: 0.0155160
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0157048, upper bound: 0.0155646

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0058694, 0.0058367
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0133038, upper bound: 0.0131571
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0133038, upper bound: 0.0131571
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0058493, 0.0058607
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0133038, upper bound: 0.0131571
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0133038, upper bound: 0.0131571
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0060721, 0.0058600
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153395, upper bound: 0.0153498
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155187, upper bound: 0.0153498
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0060325, 0.0058711
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155115, upper bound: 0.0153498
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155115, upper bound: 0.0153498
time: 1.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0060292
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 231

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162770, upper bound: 0.0161979
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162770, upper bound: 0.0163289
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0060260
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164519, upper bound: 0.0163525
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164533, upper bound: 0.0163497
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153833, upper bound: 0.0155226
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153833, upper bound: 0.0155226
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153586, upper bound: 0.0152881
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153586, upper bound: 0.0152881
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128029, upper bound: 0.0128480
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128029, upper bound: 0.0128480
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163139, upper bound: 0.0162804
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162253, upper bound: 0.0163429
time: 2.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0148847, upper bound: 0.0151164
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0148846, upper bound: 0.0151164
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128652, upper bound: 0.0129668
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128652, upper bound: 0.0129668
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0154497, upper bound: 0.0155990
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155641, upper bound: 0.0155913
time: 2.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0060992, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0154300, upper bound: 0.0154396
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153776, upper bound: 0.0154742
time: 1.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0140754, upper bound: 0.0139399
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0140754, upper bound: 0.0139399
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0154188, upper bound: 0.0153774
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153708, upper bound: 0.0154055
time: 1.89 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 6.81 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0133038, upper bound: 0.0131571
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0133038, upper bound: 0.0131571
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0133038, upper bound: 0.0131571
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0133038, upper bound: 0.0131571
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0153395, upper bound: 0.0153498
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0155187, upper bound: 0.0153498
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0155115, upper bound: 0.0153498
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0155115, upper bound: 0.0153498
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0162770, upper bound: 0.0161979
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0162770, upper bound: 0.0163289
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0164519, upper bound: 0.0163525
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0164533, upper bound: 0.0163497
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0153833, upper bound: 0.0155226
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0153833, upper bound: 0.0155226
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0153586, upper bound: 0.0152881
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0153586, upper bound: 0.0152881
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0128029, upper bound: 0.0128480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0128029, upper bound: 0.0128480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0163139, upper bound: 0.0162804
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0162253, upper bound: 0.0163429
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0148847, upper bound: 0.0151164
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0148846, upper bound: 0.0151164
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0128652, upper bound: 0.0129668
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0128652, upper bound: 0.0129668
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0154497, upper bound: 0.0155990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0155641, upper bound: 0.0155913
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0154300, upper bound: 0.0154396
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0153776, upper bound: 0.0154742
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0140754, upper bound: 0.0139399
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0140754, upper bound: 0.0139399
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0154188, upper bound: 0.0153774
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.81
Output dim: 0, lower bound: -0.0153708, upper bound: 0.0154055

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0060652, 0.0058331
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145452, upper bound: 0.0143142
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145452, upper bound: 0.0143142
time: 1.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0059521, 0.0059251
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0130345, upper bound: 0.0131352
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0130345, upper bound: 0.0131352
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0060311
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 231

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164115, upper bound: 0.0161763
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162477, upper bound: 0.0163124
time: 1.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0060306
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153149, upper bound: 0.0152333
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0153149, upper bound: 0.0152333
time: 2.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0060756, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 231

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159323, upper bound: 0.0160924
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0161341, upper bound: 0.0160093
time: 1.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0060617, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160301, upper bound: 0.0161658
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160301, upper bound: 0.0162582
time: 1.50 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 4.73 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.73
Output dim: 0, lower bound: -0.0145452, upper bound: 0.0143142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.73
Output dim: 0, lower bound: -0.0145452, upper bound: 0.0143142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.73
Output dim: 0, lower bound: -0.0130345, upper bound: 0.0131352
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.73
Output dim: 0, lower bound: -0.0130345, upper bound: 0.0131352
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 4.73
Output dim: 0, lower bound: -0.0164115, upper bound: 0.0161763
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 4.73
Output dim: 0, lower bound: -0.0162477, upper bound: 0.0163124
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.73
Output dim: 0, lower bound: -0.0153149, upper bound: 0.0152333
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.73
Output dim: 0, lower bound: -0.0153149, upper bound: 0.0152333
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 4.73
Output dim: 0, lower bound: -0.0159323, upper bound: 0.0160924
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 4.73
Output dim: 0, lower bound: -0.0161341, upper bound: 0.0160093
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 4.73
Output dim: 0, lower bound: -0.0160301, upper bound: 0.0161658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 4.73
Output dim: 0, lower bound: -0.0160301, upper bound: 0.0162582

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0060428, 0.0058177
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0150784, upper bound: 0.0148974
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0150784, upper bound: 0.0148974
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0059356, 0.0059186
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158314, upper bound: 0.0158918
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158203, upper bound: 0.0158958
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0058784, 0.0061023
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159101, upper bound: 0.0159211
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0157928, upper bound: 0.0160083
time: 1.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0059323, 0.0060622
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0147905, upper bound: 0.0146685
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0147905, upper bound: 0.0146685
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0057711, 0.0058721
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158475, upper bound: 0.0159783
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159595, upper bound: 0.0158916
time: 1.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0056611, 0.0059606
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=246
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0138836, upper bound: 0.0140783
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0138836, upper bound: 0.0140783
time: 1.54 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 4.68 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.68
Output dim: 0, lower bound: -0.0150784, upper bound: 0.0148974
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.68
Output dim: 0, lower bound: -0.0150784, upper bound: 0.0148974
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.68
Output dim: 0, lower bound: -0.0158314, upper bound: 0.0158918
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.68
Output dim: 0, lower bound: -0.0158203, upper bound: 0.0158958
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.68
Output dim: 0, lower bound: -0.0159101, upper bound: 0.0159211
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.68
Output dim: 0, lower bound: -0.0157928, upper bound: 0.0160083
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.68
Output dim: 0, lower bound: -0.0147905, upper bound: 0.0146685
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.68
Output dim: 0, lower bound: -0.0147905, upper bound: 0.0146685
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 4.68
Output dim: 0, lower bound: -0.0158475, upper bound: 0.0159783
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 4.68
Output dim: 0, lower bound: -0.0159595, upper bound: 0.0158916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 4.68
Output dim: 0, lower bound: -0.0138836, upper bound: 0.0140783
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 4.68
Output dim: 0, lower bound: -0.0138836, upper bound: 0.0140783

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0058616, 0.0057049
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0157487, upper bound: 0.0157360
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0156605, upper bound: 0.0158080
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0057218, 0.0058021
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144786, upper bound: 0.0145508
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144786, upper bound: 0.0145508
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0056182, 0.0057469
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0137103, upper bound: 0.0137544
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0137103, upper bound: 0.0137544
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0055044, 0.0058415
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144795, upper bound: 0.0146309
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144795, upper bound: 0.0146309
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0056004, 0.0057600
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0136662, upper bound: 0.0137918
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0136662, upper bound: 0.0137918
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0056627, 0.0057014
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=245
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 231
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144870, upper bound: 0.0145578
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145916, upper bound: 0.0145578
time: 2.00 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 5.20 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 5.20
Output dim: 0, lower bound: -0.0157487, upper bound: 0.0157360
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 5.20
Output dim: 0, lower bound: -0.0156605, upper bound: 0.0158080
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 5.20
Output dim: 0, lower bound: -0.0144786, upper bound: 0.0145508
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 5.20
Output dim: 0, lower bound: -0.0144786, upper bound: 0.0145508
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 5.20
Output dim: 0, lower bound: -0.0137103, upper bound: 0.0137544
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 5.20
Output dim: 0, lower bound: -0.0137103, upper bound: 0.0137544
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 5.20
Output dim: 0, lower bound: -0.0144795, upper bound: 0.0146309
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 5.20
Output dim: 0, lower bound: -0.0144795, upper bound: 0.0146309
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 5.20
Output dim: 0, lower bound: -0.0136662, upper bound: 0.0137918
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 5.20
Output dim: 0, lower bound: -0.0136662, upper bound: 0.0137918
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 5.20
Output dim: 0, lower bound: -0.0144870, upper bound: 0.0145578
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 5.20
Output dim: 0, lower bound: -0.0145916, upper bound: 0.0145578

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293
1: -0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382
2: 0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210
3: -0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0060523, 0.0060584
4: 0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575
5: 0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635
6: -0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247
7: -0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719
8: -0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123
9: -0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=244
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0135475, upper bound: 0.0136703
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0135475, upper bound: 0.0136703
time: 1.56 seconds

## Summary of splitting (split count: 12)
- Time for RS candidates: 4.60 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 13, time: 4.60
Output dim: 0, lower bound: -0.0135475, upper bound: 0.0136703
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 13, time: 4.60
Output dim: 0, lower bound: -0.0135475, upper bound: 0.0136703

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.52 + 561.76 = 565.29 seconds
