## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0005928


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0013260, 0.0013260)
1: (-0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0003304, 0.0003304)
2: (-0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0017509, 0.0017509)
3: (0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0007969, 0.0007969)
4: (-0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0003389, 0.0003389)
5: (-0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0022022, 0.0022022)
6: (0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0005589, 0.0005589)
7: (0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0014461, 0.0014461)
8: (0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0007605, 0.0007605)
9: (-0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0008818, 0.0008818)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.58 = 2.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0006832, upper bound: 0.0006832

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006343, upper bound: 0.0005747
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005747, upper bound: 0.0006343
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -0.0006343, upper bound: 0.0005747
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -0.0005747, upper bound: 0.0006343

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0010191, 0.0009736
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0002539, 0.0002426
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0012857, 0.0013457
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0006125, 0.0005852
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002488, 0.0002605
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0016171, 0.0016925
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0004296, 0.0004104
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0011115, 0.0010619
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0005845, 0.0005584
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0006475, 0.0006778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006121, upper bound: 0.0005613
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006208, upper bound: 0.0005559
time: 0.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0009736, 0.0010191
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0002426, 0.0002539
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0013457, 0.0012857
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0005852, 0.0006125
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002605, 0.0002488
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0016925, 0.0016171
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0004104, 0.0004296
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0010619, 0.0011115
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0005584, 0.0005845
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0006778, 0.0006475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005559, upper bound: 0.0006208
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005613, upper bound: 0.0006121
time: 0.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 0, lower bound: -0.0006121, upper bound: 0.0005613
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 0, lower bound: -0.0006208, upper bound: 0.0005559
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 0, lower bound: -0.0005559, upper bound: 0.0006208
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 0, lower bound: -0.0005613, upper bound: 0.0006121

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0008289, 0.0007911
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0002065, 0.0001971
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0010447, 0.0010946
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0004982, 0.0004755
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002022, 0.0002119
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0013139, 0.0013767
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003494, 0.0003335
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0009041, 0.0008628
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004754, 0.0004538
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0005262, 0.0005513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005986, upper bound: 0.0005402
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005873, upper bound: 0.0005477
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0008376, 0.0007835
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0002087, 0.0001952
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0010346, 0.0011060
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0005034, 0.0004709
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002002, 0.0002141
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0013012, 0.0013910
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003531, 0.0003303
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0009135, 0.0008545
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004804, 0.0004494
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0005211, 0.0005570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006153, upper bound: 0.0005491
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006062, upper bound: 0.0005498
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0007835, 0.0008376
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0001952, 0.0002087
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0011060, 0.0010346
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0004709, 0.0005034
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002141, 0.0002002
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0013910, 0.0013012
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003303, 0.0003531
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0008545, 0.0009135
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004494, 0.0004804
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0005570, 0.0005211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005102, upper bound: 0.0005735
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005095, upper bound: 0.0005764
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0007911, 0.0008289
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0001971, 0.0002065
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0010946, 0.0010447
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0004755, 0.0004982
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002119, 0.0002022
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0013767, 0.0013139
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003335, 0.0003494
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0008628, 0.0009041
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004538, 0.0004754
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0005513, 0.0005262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005611, upper bound: 0.0005532
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005133, upper bound: 0.0006120
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0005986, upper bound: 0.0005402
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0005873, upper bound: 0.0005477
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0006153, upper bound: 0.0005491
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0006062, upper bound: 0.0005498
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0005102, upper bound: 0.0005735
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0005095, upper bound: 0.0005764
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0005611, upper bound: 0.0005532
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -0.0005133, upper bound: 0.0006120

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0007894, 0.0007448
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0001967, 0.0001856
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0009835, 0.0010424
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0004744, 0.0004476
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0001903, 0.0002017
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0012370, 0.0013110
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003328, 0.0003140
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0008609, 0.0008123
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004528, 0.0004272
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0004953, 0.0005250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 145

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004544, upper bound: 0.0004811
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005181, upper bound: 0.0004273
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0008406, 0.0007842
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0002095, 0.0001954
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0010355, 0.0011100
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0005052, 0.0004713
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002004, 0.0002148
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0013024, 0.0013961
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003543, 0.0003306
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0009168, 0.0008553
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004821, 0.0004498
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0005215, 0.0005591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006021, upper bound: 0.0005289
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005927, upper bound: 0.0005357
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0008383, 0.0007855
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0002089, 0.0001957
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0010373, 0.0011069
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0005038, 0.0004721
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002008, 0.0002142
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0013046, 0.0013922
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003534, 0.0003311
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0009143, 0.0008567
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004808, 0.0004505
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0005224, 0.0005575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006061, upper bound: 0.0005017
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005380, upper bound: 0.0005497
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0007907, 0.0008286
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0001970, 0.0002065
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0010942, 0.0010441
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0004753, 0.0004980
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002118, 0.0002021
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0013762, 0.0013133
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003333, 0.0003493
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0008624, 0.0009037
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004535, 0.0004753
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0005511, 0.0005259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005072, upper bound: 0.0005986
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005026, upper bound: 0.0006065
time: 0.73 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0004544, upper bound: 0.0004811
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0005181, upper bound: 0.0004273
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0006021, upper bound: 0.0005289
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0005927, upper bound: 0.0005357
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0006061, upper bound: 0.0005017
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0005380, upper bound: 0.0005497
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0005072, upper bound: 0.0005986
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0005026, upper bound: 0.0006065

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0007980, 0.0007374
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0001988, 0.0001838
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0009738, 0.0010538
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0004796, 0.0004432
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0001885, 0.0002040
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0012248, 0.0013254
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003364, 0.0003109
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0008704, 0.0008043
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004577, 0.0004230
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0004905, 0.0005307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006020, upper bound: 0.0004825
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005408, upper bound: 0.0005288
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0008373, 0.0007831
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0002086, 0.0001951
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0010340, 0.0011056
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0005032, 0.0004707
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002001, 0.0002140
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0013006, 0.0013905
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003529, 0.0003301
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0009131, 0.0008541
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004802, 0.0004491
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0005208, 0.0005568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005929, upper bound: 0.0004849
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005848, upper bound: 0.0004884
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0007930, 0.0008296
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0001976, 0.0002067
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0010955, 0.0010472
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0004766, 0.0004986
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002120, 0.0002027
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0013779, 0.0013170
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003343, 0.0003497
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0008649, 0.0009048
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004548, 0.0004759
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0005518, 0.0005274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004614, upper bound: 0.0005470
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004591, upper bound: 0.0005541
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0007918, 0.0008320
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0001973, 0.0002073
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0010986, 0.0010456
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0004759, 0.0005000
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002126, 0.0002024
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0013818, 0.0013151
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003338, 0.0003507
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0008636, 0.0009074
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004542, 0.0004772
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0005533, 0.0005266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004574, upper bound: 0.0005576
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004527, upper bound: 0.0005617
time: 0.74 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.71 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0006020, upper bound: 0.0004825
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0005408, upper bound: 0.0005288
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0005929, upper bound: 0.0004849
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0005848, upper bound: 0.0004884
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0004614, upper bound: 0.0005470
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0004591, upper bound: 0.0005541
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0004574, upper bound: 0.0005576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.0004527, upper bound: 0.0005617

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0008373, 0.0007831
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0002086, 0.0001951
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0010340, 0.0011056
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0005032, 0.0004707
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0002001, 0.0002140
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0013006, 0.0013905
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003529, 0.0003301
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0009131, 0.0008541
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004802, 0.0004491
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0005208, 0.0005568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 129
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005572, upper bound: 0.0004336
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005543, upper bound: 0.0004372
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 1.0026212, 1.0046849, 1.0026212, 1.0046849, -0.0007980, 0.0007374
1: -0.0006108, -0.0000966, -0.0006108, -0.0000966, -0.0001988, 0.0001838
2: -0.0095420, -0.0068169, -0.0095420, -0.0068169, -0.0009738, 0.0010538
3: 0.0018296, 0.0030700, 0.0018296, 0.0030700, -0.0004796, 0.0004432
4: -0.0013190, -0.0007915, -0.0013190, -0.0007915, -0.0001885, 0.0002040
5: -0.0130419, -0.0096144, -0.0130419, -0.0096144, -0.0012248, 0.0013254
6: 0.0039811, 0.0048510, 0.0039811, 0.0048510, -0.0003364, 0.0003109
7: 0.0071626, 0.0094134, 0.0071626, 0.0094134, -0.0008704, 0.0008043
8: 0.0042026, 0.0053863, 0.0042026, 0.0053863, -0.0004577, 0.0004230
9: -0.0081095, -0.0067370, -0.0081095, -0.0067370, -0.0004905, 0.0005307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 145
type: RSZ, layer: 3, pos: 144
type: RSZ, layer: 3, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 242

### Candidate
type: RSZ, layer: 3, pos: 145

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004543, upper bound: 0.0004351
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0005152, upper bound: 0.0003994
time: 0.78 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0005572, upper bound: 0.0004336
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0005543, upper bound: 0.0004372
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0004543, upper bound: 0.0004351
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0005152, upper bound: 0.0003994

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.84 + 44.95 = 47.79 seconds
