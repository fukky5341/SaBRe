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
Threshold: 0.0013


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0028236, 0.0028236)
1: (-0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0007036, 0.0007036)
2: (0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0037285, 0.0037285)
3: (-0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016971, 0.0016971)
4: (0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007216, 0.0007216)
5: (0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0046895, 0.0046895)
6: (-0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011902, 0.0011902)
7: (-0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0030795, 0.0030795)
8: (-0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0016195, 0.0016195)
9: (-0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018779, 0.0018779)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 1.72 = 3.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0016084, upper bound: 0.0016085

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015310, upper bound: 0.0015121
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015121, upper bound: 0.0015310
time: 0.82 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -0.0015310, upper bound: 0.0015121
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -0.0015121, upper bound: 0.0015310

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0028247, 0.0027711
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0007038, 0.0006905
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036592, 0.0037300
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016977, 0.0016655
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007082, 0.0007219
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0046023, 0.0046914
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011907, 0.0011681
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0030808, 0.0030223
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0016201, 0.0015894
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018430, 0.0018786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014921, upper bound: 0.0014777
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014908, upper bound: 0.0014797
time: 0.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027711, 0.0028236
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006905, 0.0007036
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0037285, 0.0036592
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016655, 0.0016971
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007216, 0.0007082
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0046895, 0.0046023
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011681, 0.0011902
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0030223, 0.0030795
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015894, 0.0016195
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018779, 0.0018430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014797, upper bound: 0.0014908
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014777, upper bound: 0.0014921
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 0, lower bound: -0.0014921, upper bound: 0.0014777
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 0, lower bound: -0.0014908, upper bound: 0.0014797
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 0, lower bound: -0.0014797, upper bound: 0.0014908
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 0, lower bound: -0.0014777, upper bound: 0.0014921

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0028098, 0.0027545
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0007001, 0.0006863
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036373, 0.0037103
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016888, 0.0016555
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007040, 0.0007181
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045747, 0.0046666
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011844, 0.0011611
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0030645, 0.0030042
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0016116, 0.0015799
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018319, 0.0018687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014344, upper bound: 0.0014310
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014371, upper bound: 0.0014272
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0028081, 0.0027557
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006997, 0.0006867
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036389, 0.0037081
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016878, 0.0016563
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007043, 0.0007177
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045768, 0.0046638
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011837, 0.0011616
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0030627, 0.0030055
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0016106, 0.0015806
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018327, 0.0018676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014336, upper bound: 0.0014317
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014366, upper bound: 0.0014284
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027557, 0.0028072
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006867, 0.0006995
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0037069, 0.0036389
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016563, 0.0016872
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007175, 0.0007043
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0046623, 0.0045768
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011616, 0.0011833
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0030055, 0.0030617
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015806, 0.0016101
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018670, 0.0018327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014284, upper bound: 0.0014366
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014317, upper bound: 0.0014336
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027545, 0.0028085
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006863, 0.0006998
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0037085, 0.0036373
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016555, 0.0016880
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007178, 0.0007040
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0046644, 0.0045747
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011611, 0.0011839
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0030042, 0.0030630
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015799, 0.0016108
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018678, 0.0018319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0014371
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014310, upper bound: 0.0014344
time: 0.75 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0014344, upper bound: 0.0014310
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0014371, upper bound: 0.0014272
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0014336, upper bound: 0.0014317
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0014366, upper bound: 0.0014284
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0014284, upper bound: 0.0014366
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0014317, upper bound: 0.0014336
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0014371
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0014310, upper bound: 0.0014344

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027726, 0.0027217
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006908, 0.0006782
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035939, 0.0036611
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016664, 0.0016358
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006956, 0.0007086
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045202, 0.0046048
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011687, 0.0011473
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0030239, 0.0029684
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015902, 0.0015610
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018101, 0.0018439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013858, upper bound: 0.0013916
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013955, upper bound: 0.0013797
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027765, 0.0027172
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006918, 0.0006771
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035881, 0.0036663
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016688, 0.0016331
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006945, 0.0007096
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045129, 0.0046113
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011704, 0.0011454
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0030282, 0.0029635
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015925, 0.0015585
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018071, 0.0018466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013858, upper bound: 0.0013896
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013963, upper bound: 0.0013796
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027709, 0.0027235
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006904, 0.0006786
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035964, 0.0036589
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016654, 0.0016369
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006961, 0.0007082
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045233, 0.0046019
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011680, 0.0011481
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0030220, 0.0029704
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015893, 0.0015621
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018113, 0.0018428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013821, upper bound: 0.0013929
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013944, upper bound: 0.0013835
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027744, 0.0027185
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006913, 0.0006774
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035897, 0.0036636
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016675, 0.0016339
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006948, 0.0007091
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045149, 0.0046079
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011695, 0.0011459
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0030259, 0.0029649
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015913, 0.0015592
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018080, 0.0018452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013822, upper bound: 0.0013915
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013957, upper bound: 0.0013834
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027185, 0.0027744
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006774, 0.0006913
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036636, 0.0035897
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016339, 0.0016675
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007091, 0.0006948
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0046078, 0.0045149
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011459, 0.0011695
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029649, 0.0030259
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015592, 0.0015913
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018452, 0.0018080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013834, upper bound: 0.0013957
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0013822
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027235, 0.0027700
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006786, 0.0006902
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036577, 0.0035964
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016369, 0.0016648
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007079, 0.0006961
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0046004, 0.0045233
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011481, 0.0011676
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029704, 0.0030210
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015621, 0.0015887
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018422, 0.0018113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013835, upper bound: 0.0013944
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013929, upper bound: 0.0013821
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027172, 0.0027763
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006771, 0.0006918
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036660, 0.0035881
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016331, 0.0016686
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007096, 0.0006945
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0046109, 0.0045128
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011454, 0.0011703
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029635, 0.0030279
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015585, 0.0015924
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018464, 0.0018071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013796, upper bound: 0.0013964
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013896, upper bound: 0.0013858
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027217, 0.0027712
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006782, 0.0006905
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036594, 0.0035939
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016358, 0.0016656
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007083, 0.0006956
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0046025, 0.0045202
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011473, 0.0011682
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029684, 0.0030224
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015610, 0.0015895
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018430, 0.0018101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013797, upper bound: 0.0013955
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013916, upper bound: 0.0013858
time: 0.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013858, upper bound: 0.0013916
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013955, upper bound: 0.0013797
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013858, upper bound: 0.0013896
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013963, upper bound: 0.0013796
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013821, upper bound: 0.0013929
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013944, upper bound: 0.0013835
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013822, upper bound: 0.0013915
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013957, upper bound: 0.0013834
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013834, upper bound: 0.0013957
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0013822
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013835, upper bound: 0.0013944
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013929, upper bound: 0.0013821
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013796, upper bound: 0.0013964
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013896, upper bound: 0.0013858
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013797, upper bound: 0.0013955
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 0, lower bound: -0.0013916, upper bound: 0.0013858

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027401, 0.0026964
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006828, 0.0006719
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035605, 0.0036183
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016469, 0.0016206
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006891, 0.0007003
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044782, 0.0045509
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011551, 0.0011366
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029885, 0.0029408
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015716, 0.0015465
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017933, 0.0018224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013304, upper bound: 0.0013674
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013598, upper bound: 0.0013192
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027473, 0.0026906
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006845, 0.0006704
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035530, 0.0036277
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016512, 0.0016172
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006877, 0.0007021
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044687, 0.0045627
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011581, 0.0011342
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029963, 0.0029345
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015757, 0.0015432
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017895, 0.0018271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013310, upper bound: 0.0013543
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013697, upper bound: 0.0013179
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027442, 0.0026919
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006838, 0.0006708
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035546, 0.0036237
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016493, 0.0016179
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006880, 0.0007014
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044708, 0.0045576
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011568, 0.0011347
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029929, 0.0029359
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015740, 0.0015440
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017903, 0.0018251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013304, upper bound: 0.0013650
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013598, upper bound: 0.0013182
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027512, 0.0026864
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006855, 0.0006694
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035473, 0.0036329
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016535, 0.0016146
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006866, 0.0007031
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044616, 0.0045692
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011597, 0.0011324
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0030005, 0.0029299
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015780, 0.0015408
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017866, 0.0018297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013310, upper bound: 0.0013541
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013712, upper bound: 0.0013173
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027384, 0.0026982
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006823, 0.0006723
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035630, 0.0036160
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016459, 0.0016217
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006896, 0.0006999
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044813, 0.0045480
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011543, 0.0011374
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029866, 0.0029428
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015706, 0.0015476
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017945, 0.0018212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013199, upper bound: 0.0013680
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013562, upper bound: 0.0013285
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027456, 0.0026924
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006841, 0.0006709
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035553, 0.0036255
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016502, 0.0016182
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006881, 0.0007017
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044717, 0.0045599
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011573, 0.0011350
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029944, 0.0029365
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015747, 0.0015443
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017906, 0.0018260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013210, upper bound: 0.0013577
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013686, upper bound: 0.0013281
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027421, 0.0026932
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006833, 0.0006711
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035563, 0.0036209
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016481, 0.0016187
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006883, 0.0007008
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044729, 0.0045541
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011559, 0.0011353
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029906, 0.0029373
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015727, 0.0015447
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017911, 0.0018237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013205, upper bound: 0.0013663
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013563, upper bound: 0.0013285
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0027491, 0.0026878
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006850, 0.0006697
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035492, 0.0036302
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016523, 0.0016155
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006869, 0.0007026
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044640, 0.0045658
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011589, 0.0011330
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029983, 0.0029315
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015768, 0.0015416
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017876, 0.0018284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013215, upper bound: 0.0013574
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013707, upper bound: 0.0013281
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026878, 0.0027521
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006697, 0.0006857
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036341, 0.0035492
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016155, 0.0016541
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007034, 0.0006869
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045707, 0.0044640
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011330, 0.0011601
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029315, 0.0030015
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015416, 0.0015785
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018303, 0.0017876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013281, upper bound: 0.0013707
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013574, upper bound: 0.0013215
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026932, 0.0027463
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006711, 0.0006843
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036265, 0.0035563
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016187, 0.0016506
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007019, 0.0006883
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045612, 0.0044729
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011353, 0.0011577
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029373, 0.0029953
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015447, 0.0015752
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018265, 0.0017911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013285, upper bound: 0.0013563
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013663, upper bound: 0.0013205
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026924, 0.0027476
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006709, 0.0006846
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036282, 0.0035553
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016182, 0.0016514
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007022, 0.0006881
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045633, 0.0044717
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011350, 0.0011582
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029365, 0.0029967
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015443, 0.0015759
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018274, 0.0017906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013281, upper bound: 0.0013686
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013577, upper bound: 0.0013210
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026982, 0.0027421
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006723, 0.0006832
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036209, 0.0035630
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016217, 0.0016481
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007008, 0.0006896
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045541, 0.0044813
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011374, 0.0011559
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029428, 0.0029906
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015476, 0.0015727
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018237, 0.0017945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013285, upper bound: 0.0013562
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013680, upper bound: 0.0013199
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026864, 0.0027539
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006694, 0.0006862
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036365, 0.0035473
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016146, 0.0016552
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007038, 0.0006866
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045738, 0.0044616
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011324, 0.0011609
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029299, 0.0030035
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015408, 0.0015795
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018315, 0.0017866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0013712
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013541, upper bound: 0.0013310
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026919, 0.0027481
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006708, 0.0006848
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036289, 0.0035546
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016179, 0.0016517
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007024, 0.0006880
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045642, 0.0044708
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011347, 0.0011584
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029359, 0.0029972
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015440, 0.0015762
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018277, 0.0017903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013182, upper bound: 0.0013598
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013650, upper bound: 0.0013304
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026906, 0.0027489
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006704, 0.0006849
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036298, 0.0035530
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016172, 0.0016521
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007025, 0.0006877
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045654, 0.0044687
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011342, 0.0011587
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029345, 0.0029980
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015432, 0.0015766
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018282, 0.0017895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013179, upper bound: 0.0013697
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013543, upper bound: 0.0013310
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026964, 0.0027435
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006719, 0.0006836
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0036228, 0.0035605
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016206, 0.0016489
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0007012, 0.0006891
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0045565, 0.0044782
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011366, 0.0011565
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029408, 0.0029922
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015465, 0.0015736
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0018246, 0.0017933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013192, upper bound: 0.0013598
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013674, upper bound: 0.0013304
time: 0.90 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013304, upper bound: 0.0013674
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013598, upper bound: 0.0013192
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013310, upper bound: 0.0013543
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013697, upper bound: 0.0013179
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013304, upper bound: 0.0013650
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013598, upper bound: 0.0013182
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013310, upper bound: 0.0013541
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013712, upper bound: 0.0013173
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013199, upper bound: 0.0013680
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013562, upper bound: 0.0013285
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013210, upper bound: 0.0013577
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013686, upper bound: 0.0013281
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013205, upper bound: 0.0013663
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013563, upper bound: 0.0013285
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013215, upper bound: 0.0013574
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013707, upper bound: 0.0013281
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013281, upper bound: 0.0013707
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013574, upper bound: 0.0013215
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013285, upper bound: 0.0013563
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013663, upper bound: 0.0013205
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013281, upper bound: 0.0013686
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013577, upper bound: 0.0013210
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013285, upper bound: 0.0013562
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013680, upper bound: 0.0013199
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013173, upper bound: 0.0013712
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013541, upper bound: 0.0013310
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013182, upper bound: 0.0013598
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013650, upper bound: 0.0013304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013179, upper bound: 0.0013697
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013543, upper bound: 0.0013310
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013192, upper bound: 0.0013598
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -0.0013674, upper bound: 0.0013304

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026468, 0.0026322
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006595, 0.0006559
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034757, 0.0034951
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015908, 0.0015820
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006727, 0.0006765
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043716, 0.0043959
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011157, 0.0011096
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028867, 0.0028707
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015181, 0.0015097
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017506, 0.0017603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012936, upper bound: 0.0013270
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012891, upper bound: 0.0013309
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026789, 0.0026030
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006675, 0.0006486
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034373, 0.0035375
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016101, 0.0015645
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006653, 0.0006847
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043232, 0.0044492
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011293, 0.0010973
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029217, 0.0028390
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015365, 0.0014930
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017312, 0.0017817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013232, upper bound: 0.0012801
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013201, upper bound: 0.0012835
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026539, 0.0026261
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006613, 0.0006544
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034677, 0.0035045
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015951, 0.0015784
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006712, 0.0006783
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043615, 0.0044077
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011187, 0.0011070
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028945, 0.0028641
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015222, 0.0015062
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017465, 0.0017650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012944, upper bound: 0.0013156
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012903, upper bound: 0.0013180
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026863, 0.0025973
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006694, 0.0006472
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034297, 0.0035473
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016146, 0.0015611
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006638, 0.0006866
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043137, 0.0044616
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011324, 0.0010949
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029298, 0.0028327
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015408, 0.0014897
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017274, 0.0017866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013332, upper bound: 0.0012775
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013300, upper bound: 0.0012816
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026509, 0.0026271
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006605, 0.0006546
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034690, 0.0035004
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015932, 0.0015789
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006714, 0.0006775
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043631, 0.0044026
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011174, 0.0011074
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028911, 0.0028652
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015204, 0.0015068
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017472, 0.0017630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012936, upper bound: 0.0013255
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012891, upper bound: 0.0013286
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026823, 0.0025986
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006684, 0.0006475
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034314, 0.0035420
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016122, 0.0015618
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006641, 0.0006855
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043158, 0.0044549
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011307, 0.0010954
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029255, 0.0028341
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015385, 0.0014904
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017282, 0.0017839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013233, upper bound: 0.0012793
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013201, upper bound: 0.0012825
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026578, 0.0026215
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006623, 0.0006532
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034617, 0.0035097
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015974, 0.0015756
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006700, 0.0006793
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043539, 0.0044142
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011204, 0.0011051
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028988, 0.0028591
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015244, 0.0015036
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017435, 0.0017676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012944, upper bound: 0.0013156
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012905, upper bound: 0.0013178
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026893, 0.0025930
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006701, 0.0006461
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034241, 0.0035512
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016163, 0.0015585
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006627, 0.0006873
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043066, 0.0044664
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011336, 0.0010931
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029330, 0.0028281
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015425, 0.0014872
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017245, 0.0017886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013345, upper bound: 0.0012775
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013307, upper bound: 0.0012813
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026451, 0.0026340
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006591, 0.0006563
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034781, 0.0034928
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015898, 0.0015831
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006732, 0.0006760
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043746, 0.0043930
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011150, 0.0011103
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028848, 0.0028727
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015171, 0.0015107
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017518, 0.0017591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012840, upper bound: 0.0013281
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012803, upper bound: 0.0013313
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026775, 0.0026049
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006672, 0.0006491
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034397, 0.0035356
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016093, 0.0015656
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006658, 0.0006843
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043263, 0.0044469
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011287, 0.0010981
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029202, 0.0028410
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015357, 0.0014941
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017324, 0.0017807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013199, upper bound: 0.0012885
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013169, upper bound: 0.0012917
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026522, 0.0026281
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006609, 0.0006549
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034704, 0.0035022
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015941, 0.0015796
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006717, 0.0006778
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043648, 0.0044049
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011180, 0.0011078
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028926, 0.0028663
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015212, 0.0015074
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017479, 0.0017639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012854, upper bound: 0.0013184
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012820, upper bound: 0.0013211
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026848, 0.0025991
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006690, 0.0006476
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034321, 0.0035452
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016136, 0.0015621
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006643, 0.0006862
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043166, 0.0044589
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011317, 0.0010956
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029281, 0.0028347
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015399, 0.0014907
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017286, 0.0017856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013322, upper bound: 0.0012872
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013289, upper bound: 0.0012913
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026487, 0.0026285
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006600, 0.0006550
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034709, 0.0034976
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015920, 0.0015798
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006718, 0.0006770
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043655, 0.0043991
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011165, 0.0011080
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028888, 0.0028668
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015192, 0.0015076
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017482, 0.0017616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012844, upper bound: 0.0013268
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012804, upper bound: 0.0013299
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026806, 0.0025998
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006679, 0.0006478
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034330, 0.0035396
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016111, 0.0015626
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006645, 0.0006851
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043179, 0.0044519
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011299, 0.0010959
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029235, 0.0028355
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015374, 0.0014911
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017291, 0.0017827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013199, upper bound: 0.0012883
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013169, upper bound: 0.0012916
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026558, 0.0026228
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006618, 0.0006535
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034633, 0.0035069
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015962, 0.0015764
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006703, 0.0006788
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043560, 0.0044108
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011195, 0.0011056
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028965, 0.0028605
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015232, 0.0015043
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017443, 0.0017663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012859, upper bound: 0.0013183
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012825, upper bound: 0.0013209
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026872, 0.0025945
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006696, 0.0006465
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034260, 0.0035485
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0016151, 0.0015594
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006631, 0.0006868
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043090, 0.0044631
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011328, 0.0010937
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0029308, 0.0028297
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015413, 0.0014881
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017255, 0.0017872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013337, upper bound: 0.0012872
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013298, upper bound: 0.0012913
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025945, 0.0026820
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006465, 0.0006683
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035415, 0.0034260
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015594, 0.0016119
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006854, 0.0006631
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044543, 0.0043090
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010937, 0.0011305
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028297, 0.0029250
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014881, 0.0015383
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017837, 0.0017255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012913, upper bound: 0.0013298
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012872, upper bound: 0.0013337
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026228, 0.0026528
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006535, 0.0006610
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035030, 0.0034633
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015764, 0.0015944
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006780, 0.0006703
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044059, 0.0043560
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011056, 0.0011183
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028605, 0.0028933
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015043, 0.0015215
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017643, 0.0017443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013209, upper bound: 0.0012825
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013183, upper bound: 0.0012859
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025998, 0.0026759
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006478, 0.0006668
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035335, 0.0034330
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015626, 0.0016083
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006839, 0.0006645
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044442, 0.0043179
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010959, 0.0011280
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028355, 0.0029184
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014911, 0.0015348
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017796, 0.0017291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012916, upper bound: 0.0013169
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012883, upper bound: 0.0013199
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026285, 0.0026471
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006550, 0.0006596
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034955, 0.0034709
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015798, 0.0015910
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006765, 0.0006718
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043964, 0.0043655
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011080, 0.0011158
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028668, 0.0028870
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015076, 0.0015183
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017605, 0.0017482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013299, upper bound: 0.0012804
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013268, upper bound: 0.0012844
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025991, 0.0026769
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006476, 0.0006670
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035348, 0.0034321
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015621, 0.0016089
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006841, 0.0006643
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044458, 0.0043166
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010956, 0.0011284
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028347, 0.0029195
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014907, 0.0015353
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017803, 0.0017286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012913, upper bound: 0.0013289
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012872, upper bound: 0.0013322
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026281, 0.0026484
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006549, 0.0006599
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034971, 0.0034704
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015796, 0.0015917
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006769, 0.0006717
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043985, 0.0043648
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011078, 0.0011164
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028663, 0.0028884
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015074, 0.0015190
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017613, 0.0017479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013211, upper bound: 0.0012820
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013184, upper bound: 0.0012854
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026049, 0.0026713
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006491, 0.0006656
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035274, 0.0034397
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015656, 0.0016055
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006827, 0.0006658
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044366, 0.0043263
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010981, 0.0011260
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028410, 0.0029134
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014941, 0.0015321
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017766, 0.0017324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012917, upper bound: 0.0013169
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012885, upper bound: 0.0013199
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026340, 0.0026428
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006563, 0.0006585
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034898, 0.0034781
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015831, 0.0015884
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006754, 0.0006732
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043893, 0.0043746
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011103, 0.0011140
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028727, 0.0028824
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015107, 0.0015158
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017577, 0.0017518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013313, upper bound: 0.0012803
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013280, upper bound: 0.0012840
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025930, 0.0026837
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006461, 0.0006687
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035439, 0.0034241
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015585, 0.0016130
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006859, 0.0006627
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044572, 0.0043066
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010931, 0.0011313
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028281, 0.0029270
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014872, 0.0015393
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017849, 0.0017245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012812, upper bound: 0.0013307
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012775, upper bound: 0.0013345
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026215, 0.0026547
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006532, 0.0006615
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035055, 0.0034617
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015756, 0.0015955
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006785, 0.0006700
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044090, 0.0043539
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011051, 0.0011190
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028591, 0.0028953
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015036, 0.0015226
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017655, 0.0017435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013178, upper bound: 0.0012905
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013156, upper bound: 0.0012944
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025986, 0.0026779
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006475, 0.0006673
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035361, 0.0034314
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015618, 0.0016095
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006844, 0.0006641
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044475, 0.0043158
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010954, 0.0011288
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028341, 0.0029206
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014904, 0.0015359
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017810, 0.0017282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012825, upper bound: 0.0013201
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012793, upper bound: 0.0013233
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026271, 0.0026489
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006546, 0.0006600
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034978, 0.0034690
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015789, 0.0015921
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006770, 0.0006714
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043993, 0.0043631
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011074, 0.0011166
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028652, 0.0028890
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015068, 0.0015193
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017617, 0.0017472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013286, upper bound: 0.0012891
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013255, upper bound: 0.0012936
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025973, 0.0026783
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006472, 0.0006674
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035367, 0.0034297
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015611, 0.0016097
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006845, 0.0006638
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044482, 0.0043137
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010949, 0.0011290
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028327, 0.0029211
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014897, 0.0015362
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017813, 0.0017274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012816, upper bound: 0.0013300
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012775, upper bound: 0.0013332
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026261, 0.0026496
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006544, 0.0006602
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034988, 0.0034677
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015784, 0.0015925
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006772, 0.0006712
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044005, 0.0043615
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011070, 0.0011169
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028641, 0.0028898
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015062, 0.0015197
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017622, 0.0017465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013180, upper bound: 0.0012903
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013156, upper bound: 0.0012944
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026030, 0.0026726
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006486, 0.0006659
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0035291, 0.0034373
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015645, 0.0016063
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006830, 0.0006653
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0044387, 0.0043232
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010973, 0.0011266
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028390, 0.0029148
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014930, 0.0015329
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017774, 0.0017312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012835, upper bound: 0.0013201
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012801, upper bound: 0.0013232
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026322, 0.0026443
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006559, 0.0006589
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034917, 0.0034757
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015820, 0.0015893
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006758, 0.0006727
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043917, 0.0043716
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011096, 0.0011147
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028707, 0.0028840
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015097, 0.0015166
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017586, 0.0017506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013309, upper bound: 0.0012891
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013270, upper bound: 0.0012936
time: 0.86 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012936, upper bound: 0.0013270
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012891, upper bound: 0.0013309
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013232, upper bound: 0.0012801
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013201, upper bound: 0.0012835
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012944, upper bound: 0.0013156
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012903, upper bound: 0.0013180
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013332, upper bound: 0.0012775
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013300, upper bound: 0.0012816
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012936, upper bound: 0.0013255
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012891, upper bound: 0.0013286
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013233, upper bound: 0.0012793
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013201, upper bound: 0.0012825
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012944, upper bound: 0.0013156
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012905, upper bound: 0.0013178
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013345, upper bound: 0.0012775
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013307, upper bound: 0.0012813
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012840, upper bound: 0.0013281
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012803, upper bound: 0.0013313
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013199, upper bound: 0.0012885
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013169, upper bound: 0.0012917
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012854, upper bound: 0.0013184
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012820, upper bound: 0.0013211
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013322, upper bound: 0.0012872
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013289, upper bound: 0.0012913
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012844, upper bound: 0.0013268
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012804, upper bound: 0.0013299
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013199, upper bound: 0.0012883
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013169, upper bound: 0.0012916
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012859, upper bound: 0.0013183
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012825, upper bound: 0.0013209
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013337, upper bound: 0.0012872
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013298, upper bound: 0.0012913
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012913, upper bound: 0.0013298
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012872, upper bound: 0.0013337
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013209, upper bound: 0.0012825
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013183, upper bound: 0.0012859
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012916, upper bound: 0.0013169
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012883, upper bound: 0.0013199
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013299, upper bound: 0.0012804
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013268, upper bound: 0.0012844
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012913, upper bound: 0.0013289
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012872, upper bound: 0.0013322
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013211, upper bound: 0.0012820
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013184, upper bound: 0.0012854
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012917, upper bound: 0.0013169
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012885, upper bound: 0.0013199
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013313, upper bound: 0.0012803
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013280, upper bound: 0.0012840
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012812, upper bound: 0.0013307
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012775, upper bound: 0.0013345
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013178, upper bound: 0.0012905
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013156, upper bound: 0.0012944
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012825, upper bound: 0.0013201
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012793, upper bound: 0.0013233
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013286, upper bound: 0.0012891
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013255, upper bound: 0.0012936
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012816, upper bound: 0.0013300
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012775, upper bound: 0.0013332
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013180, upper bound: 0.0012903
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013156, upper bound: 0.0012944
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012835, upper bound: 0.0013201
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0012801, upper bound: 0.0013232
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013309, upper bound: 0.0012891
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -0.0013270, upper bound: 0.0012936

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025842, 0.0025613
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006439, 0.0006382
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033822, 0.0034125
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015532, 0.0015394
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006546, 0.0006605
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042539, 0.0042920
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010893, 0.0010797
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028185, 0.0027935
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014822, 0.0014691
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017034, 0.0017187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012404, upper bound: 0.0012709
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012346, upper bound: 0.0012720
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025753, 0.0025696
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006417, 0.0006403
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033931, 0.0034007
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015478, 0.0015444
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006567, 0.0006582
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042677, 0.0042772
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010856, 0.0010832
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028087, 0.0028025
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014771, 0.0014738
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017090, 0.0017128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012352, upper bound: 0.0012748
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012317, upper bound: 0.0012756
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026164, 0.0025313
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006519, 0.0006307
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033425, 0.0034549
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015725, 0.0015214
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006469, 0.0006687
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042040, 0.0043453
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011029, 0.0010670
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028535, 0.0027607
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015006, 0.0014518
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016835, 0.0017401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012701, upper bound: 0.0012240
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012652, upper bound: 0.0012258
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026089, 0.0025405
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006501, 0.0006330
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033547, 0.0034451
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015680, 0.0015269
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006493, 0.0006668
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042193, 0.0043330
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010998, 0.0010709
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028454, 0.0027707
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014964, 0.0014571
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016896, 0.0017351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012670, upper bound: 0.0012269
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012629, upper bound: 0.0012295
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025914, 0.0025546
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006457, 0.0006365
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033733, 0.0034219
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015575, 0.0015354
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006529, 0.0006623
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042428, 0.0043038
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010924, 0.0010769
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028262, 0.0027862
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014863, 0.0014652
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016990, 0.0017234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012415, upper bound: 0.0012595
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012350, upper bound: 0.0012612
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025826, 0.0025635
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006435, 0.0006388
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033851, 0.0034102
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015522, 0.0015408
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006552, 0.0006600
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042576, 0.0042892
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010886, 0.0010806
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028167, 0.0027959
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014812, 0.0014703
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017049, 0.0017176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012365, upper bound: 0.0012618
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012327, upper bound: 0.0012637
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026238, 0.0025256
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006538, 0.0006293
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033350, 0.0034647
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015770, 0.0015179
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006455, 0.0006706
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0041945, 0.0043577
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011060, 0.0010646
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028616, 0.0027545
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015049, 0.0014486
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016797, 0.0017450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012808, upper bound: 0.0012215
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012755, upper bound: 0.0012233
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026159, 0.0025348
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006518, 0.0006316
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033471, 0.0034542
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015722, 0.0015235
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006478, 0.0006686
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042098, 0.0043445
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011027, 0.0010685
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028530, 0.0027645
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015004, 0.0014538
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016858, 0.0017397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012771, upper bound: 0.0012256
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012728, upper bound: 0.0012276
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025883, 0.0025561
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006449, 0.0006369
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033752, 0.0034178
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015556, 0.0015363
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006533, 0.0006615
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042452, 0.0042987
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010911, 0.0010775
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028229, 0.0027877
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014845, 0.0014660
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017000, 0.0017214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012404, upper bound: 0.0012693
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012348, upper bound: 0.0012711
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025793, 0.0025645
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006427, 0.0006390
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033864, 0.0034060
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015502, 0.0015414
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006554, 0.0006592
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042592, 0.0042838
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010873, 0.0010810
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028131, 0.0027970
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014794, 0.0014709
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017056, 0.0017154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012352, upper bound: 0.0012728
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012318, upper bound: 0.0012740
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026198, 0.0025277
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006528, 0.0006298
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033378, 0.0034594
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015746, 0.0015192
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006460, 0.0006696
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0041981, 0.0043510
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011043, 0.0010655
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028573, 0.0027568
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015026, 0.0014498
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016811, 0.0017423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012701, upper bound: 0.0012233
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012652, upper bound: 0.0012252
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026121, 0.0025360
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006509, 0.0006319
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033488, 0.0034492
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015699, 0.0015242
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006482, 0.0006676
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042119, 0.0043382
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011011, 0.0010690
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028489, 0.0027659
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014982, 0.0014546
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016866, 0.0017372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012670, upper bound: 0.0012261
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012629, upper bound: 0.0012288
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025953, 0.0025501
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006467, 0.0006354
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033674, 0.0034271
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015598, 0.0015327
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006518, 0.0006633
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042353, 0.0043103
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010940, 0.0010750
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028305, 0.0027813
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014885, 0.0014627
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016960, 0.0017260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012415, upper bound: 0.0012594
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012358, upper bound: 0.0012611
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025864, 0.0025590
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006445, 0.0006376
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033791, 0.0034153
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015545, 0.0015380
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006540, 0.0006610
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042500, 0.0042956
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010903, 0.0010787
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028209, 0.0027909
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014835, 0.0014677
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017019, 0.0017201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012366, upper bound: 0.0012616
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012636
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026267, 0.0025225
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006545, 0.0006285
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033309, 0.0034686
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015787, 0.0015161
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006447, 0.0006713
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0041894, 0.0043625
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011073, 0.0010633
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028648, 0.0027511
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015066, 0.0014468
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016776, 0.0017469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012817, upper bound: 0.0012215
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012771, upper bound: 0.0012233
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026189, 0.0025305
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006525, 0.0006305
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033415, 0.0034582
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015740, 0.0015209
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006467, 0.0006693
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042027, 0.0043495
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011039, 0.0010667
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028562, 0.0027598
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015021, 0.0014514
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016829, 0.0017417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012775, upper bound: 0.0012250
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012740, upper bound: 0.0012273
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025825, 0.0025632
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006435, 0.0006387
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033846, 0.0034102
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015522, 0.0015405
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006551, 0.0006600
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042570, 0.0042891
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010886, 0.0010805
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028166, 0.0027955
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014812, 0.0014701
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017047, 0.0017175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012304, upper bound: 0.0012720
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012267, upper bound: 0.0012737
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025742, 0.0025714
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006414, 0.0006407
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033955, 0.0033991
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015471, 0.0015455
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006572, 0.0006579
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042707, 0.0042752
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010851, 0.0010839
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028075, 0.0028045
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014764, 0.0014749
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017102, 0.0017120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012267, upper bound: 0.0012752
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012234, upper bound: 0.0012771
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026150, 0.0025329
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006516, 0.0006311
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033446, 0.0034530
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015717, 0.0015223
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006473, 0.0006683
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042066, 0.0043430
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011023, 0.0010677
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028520, 0.0027624
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014998, 0.0014527
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016845, 0.0017391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012668, upper bound: 0.0012318
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012624, upper bound: 0.0012338
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026077, 0.0025423
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006498, 0.0006335
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033571, 0.0034435
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015673, 0.0015280
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006498, 0.0006665
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042224, 0.0043310
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010993, 0.0010717
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028441, 0.0027728
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014957, 0.0014582
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016908, 0.0017343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012634, upper bound: 0.0012339
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012603, upper bound: 0.0012380
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025897, 0.0025564
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006453, 0.0006370
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033757, 0.0034196
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015565, 0.0015365
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006534, 0.0006619
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042458, 0.0043010
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010916, 0.0010776
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028244, 0.0027881
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014853, 0.0014662
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017002, 0.0017223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012324, upper bound: 0.0012620
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012276, upper bound: 0.0012649
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025811, 0.0025656
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006431, 0.0006393
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033878, 0.0034084
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015513, 0.0015420
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006557, 0.0006597
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042610, 0.0042868
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010880, 0.0010815
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028151, 0.0027981
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014804, 0.0014715
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017063, 0.0017166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012285, upper bound: 0.0012641
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012251, upper bound: 0.0012676
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026222, 0.0025273
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006534, 0.0006297
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033373, 0.0034626
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015760, 0.0015190
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006459, 0.0006702
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0041974, 0.0043551
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011054, 0.0010654
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028599, 0.0027564
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015040, 0.0014496
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016808, 0.0017440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012797, upper bound: 0.0012304
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012747, upper bound: 0.0012327
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026147, 0.0025365
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006515, 0.0006320
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033495, 0.0034527
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015715, 0.0015245
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006483, 0.0006683
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042128, 0.0043426
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011022, 0.0010692
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028517, 0.0027665
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014997, 0.0014549
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016870, 0.0017389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012763, upper bound: 0.0012333
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012715, upper bound: 0.0012376
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025862, 0.0025572
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006444, 0.0006372
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033767, 0.0034150
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015544, 0.0015369
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006536, 0.0006610
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042470, 0.0042952
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010902, 0.0010779
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028206, 0.0027890
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014833, 0.0014667
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017007, 0.0017200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012306, upper bound: 0.0012705
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012271, upper bound: 0.0012728
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025774, 0.0025660
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006422, 0.0006394
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033883, 0.0034034
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015491, 0.0015422
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006558, 0.0006587
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042617, 0.0042806
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010865, 0.0010817
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028110, 0.0027986
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014783, 0.0014717
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017066, 0.0017141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012267, upper bound: 0.0012739
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012234, upper bound: 0.0012759
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026180, 0.0025286
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006523, 0.0006301
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033390, 0.0034570
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015735, 0.0015198
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006463, 0.0006691
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0041996, 0.0043480
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011036, 0.0010659
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028553, 0.0027578
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015016, 0.0014503
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016817, 0.0017411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012668, upper bound: 0.0012311
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012625, upper bound: 0.0012338
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026108, 0.0025373
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006505, 0.0006322
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033504, 0.0034475
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015691, 0.0015250
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006485, 0.0006673
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042140, 0.0043360
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011005, 0.0010695
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028474, 0.0027672
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014974, 0.0014553
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016875, 0.0017363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012634, upper bound: 0.0012337
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012603, upper bound: 0.0012379
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025932, 0.0025513
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006462, 0.0006357
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033689, 0.0034243
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015586, 0.0015334
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006520, 0.0006628
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042372, 0.0043069
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010931, 0.0010754
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028283, 0.0027825
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014874, 0.0014633
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016968, 0.0017247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012324, upper bound: 0.0012620
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012281, upper bound: 0.0012649
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025842, 0.0025602
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006439, 0.0006379
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033807, 0.0034123
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015531, 0.0015388
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006543, 0.0006605
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042521, 0.0042918
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010893, 0.0010792
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028184, 0.0027923
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014822, 0.0014684
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017027, 0.0017186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012287, upper bound: 0.0012641
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012257, upper bound: 0.0012676
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026247, 0.0025234
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006540, 0.0006288
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033322, 0.0034659
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015775, 0.0015167
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006449, 0.0006708
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0041910, 0.0043592
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011064, 0.0010637
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028626, 0.0027522
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015054, 0.0014473
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016783, 0.0017456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012812, upper bound: 0.0012303
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012766, upper bound: 0.0012327
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0026174, 0.0025319
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006522, 0.0006309
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033434, 0.0034562
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015731, 0.0015218
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006471, 0.0006689
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042051, 0.0043470
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0011033, 0.0010673
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028546, 0.0027614
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0015012, 0.0014522
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0016839, 0.0017407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012765, upper bound: 0.0012332
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012733, upper bound: 0.0012376
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025319, 0.0026123
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006309, 0.0006509
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034495, 0.0033434
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015218, 0.0015701
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006677, 0.0006471
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043386, 0.0042051
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010673, 0.0011012
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027614, 0.0028491
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014522, 0.0014983
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017374, 0.0016839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012376, upper bound: 0.0012733
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012332, upper bound: 0.0012765
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025234, 0.0026206
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006288, 0.0006530
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034605, 0.0033322
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015167, 0.0015751
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006698, 0.0006449
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043524, 0.0041910
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010637, 0.0011047
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027522, 0.0028582
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014473, 0.0015031
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017429, 0.0016783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012327, upper bound: 0.0012766
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012303, upper bound: 0.0012812
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025602, 0.0025823
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006379, 0.0006434
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034099, 0.0033807
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015388, 0.0015520
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006600, 0.0006543
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042887, 0.0042521
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010792, 0.0010885
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027923, 0.0028164
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014684, 0.0014811
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017174, 0.0017027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012676, upper bound: 0.0012257
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012641, upper bound: 0.0012287
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025513, 0.0025915
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006357, 0.0006457
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034220, 0.0033689
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015334, 0.0015576
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006623, 0.0006520
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043040, 0.0042372
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010754, 0.0010924
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027825, 0.0028264
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014633, 0.0014864
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017235, 0.0016968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012649, upper bound: 0.0012281
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012620, upper bound: 0.0012324
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025373, 0.0026056
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006322, 0.0006493
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034407, 0.0033504
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015250, 0.0015661
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006659, 0.0006485
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043275, 0.0042140
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010695, 0.0010984
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027672, 0.0028418
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014553, 0.0014945
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017329, 0.0016875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012379, upper bound: 0.0012603
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012337, upper bound: 0.0012635
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025286, 0.0026146
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006301, 0.0006515
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034525, 0.0033390
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015198, 0.0015714
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006682, 0.0006463
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043423, 0.0041996
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010659, 0.0011021
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027578, 0.0028515
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014503, 0.0014996
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017389, 0.0016817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012338, upper bound: 0.0012625
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012311, upper bound: 0.0012668
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025660, 0.0025766
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006394, 0.0006420
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034024, 0.0033883
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015422, 0.0015486
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006585, 0.0006558
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042793, 0.0042617
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010817, 0.0010861
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027986, 0.0028101
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014717, 0.0014778
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017136, 0.0017066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012759, upper bound: 0.0012234
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012739, upper bound: 0.0012267
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025572, 0.0025858
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006372, 0.0006443
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034145, 0.0033767
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015369, 0.0015541
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006609, 0.0006536
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042945, 0.0042470
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010779, 0.0010900
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027890, 0.0028202
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014667, 0.0014831
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017197, 0.0017007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012728, upper bound: 0.0012271
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012705, upper bound: 0.0012305
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025365, 0.0026071
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006320, 0.0006496
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034426, 0.0033495
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015245, 0.0015669
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006663, 0.0006483
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043299, 0.0042128
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010692, 0.0010990
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027665, 0.0028434
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014549, 0.0014953
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017339, 0.0016870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012376, upper bound: 0.0012715
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012333, upper bound: 0.0012763
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025273, 0.0026155
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006297, 0.0006517
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034538, 0.0033373
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015190, 0.0015720
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006685, 0.0006459
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043440, 0.0041974
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010654, 0.0011025
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027564, 0.0028526
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014496, 0.0015002
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017395, 0.0016808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012327, upper bound: 0.0012747
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012304, upper bound: 0.0012797
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025656, 0.0025787
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006393, 0.0006426
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034052, 0.0033878
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015420, 0.0015499
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006591, 0.0006557
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042828, 0.0042610
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010815, 0.0010870
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027981, 0.0028125
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014715, 0.0014791
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017150, 0.0017063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012676, upper bound: 0.0012251
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012641, upper bound: 0.0012285
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025564, 0.0025870
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006370, 0.0006446
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034162, 0.0033757
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015365, 0.0015549
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006612, 0.0006534
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042966, 0.0042458
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010776, 0.0010905
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027881, 0.0028215
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014662, 0.0014838
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017206, 0.0017002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012649, upper bound: 0.0012276
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012620, upper bound: 0.0012324
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025423, 0.0026012
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006335, 0.0006481
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034348, 0.0033571
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015280, 0.0015634
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006648, 0.0006498
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043201, 0.0042224
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010717, 0.0010965
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027728, 0.0028369
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014582, 0.0014919
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017299, 0.0016908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012380, upper bound: 0.0012603
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012339, upper bound: 0.0012634
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025329, 0.0026100
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006311, 0.0006503
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034464, 0.0033446
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015223, 0.0015687
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006671, 0.0006473
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043347, 0.0042066
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010677, 0.0011002
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027624, 0.0028465
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014527, 0.0014970
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017358, 0.0016845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012338, upper bound: 0.0012625
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012318, upper bound: 0.0012668
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025714, 0.0025735
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006407, 0.0006412
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033983, 0.0033955
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015455, 0.0015467
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006577, 0.0006572
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042741, 0.0042707
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010839, 0.0010848
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028045, 0.0028068
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014749, 0.0014761
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017116, 0.0017102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012771, upper bound: 0.0012234
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012752, upper bound: 0.0012267
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025632, 0.0025815
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006387, 0.0006432
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034088, 0.0033846
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015405, 0.0015516
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006598, 0.0006551
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042874, 0.0042570
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010805, 0.0010882
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027955, 0.0028155
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014701, 0.0014806
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017169, 0.0017047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012737, upper bound: 0.0012267
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012720, upper bound: 0.0012304
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025305, 0.0026142
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006305, 0.0006514
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034520, 0.0033415
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015209, 0.0015712
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006681, 0.0006467
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043417, 0.0042027
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010667, 0.0011020
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027598, 0.0028511
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014514, 0.0014994
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017386, 0.0016829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012273, upper bound: 0.0012740
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012250, upper bound: 0.0012775
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025225, 0.0026224
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006285, 0.0006534
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034629, 0.0033309
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015161, 0.0015762
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006702, 0.0006447
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043554, 0.0041894
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010633, 0.0011054
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027511, 0.0028601
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014468, 0.0015041
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017441, 0.0016776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012233, upper bound: 0.0012771
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012215, upper bound: 0.0012817
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025590, 0.0025839
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006376, 0.0006438
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034120, 0.0033791
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015380, 0.0015530
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006604, 0.0006540
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042914, 0.0042500
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010787, 0.0010892
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027909, 0.0028181
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014677, 0.0014820
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017185, 0.0017019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012636, upper bound: 0.0012334
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012616, upper bound: 0.0012366
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025501, 0.0025934
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006354, 0.0006462
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034245, 0.0033674
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015327, 0.0015587
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006628, 0.0006518
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043071, 0.0042353
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010750, 0.0010932
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027813, 0.0028284
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014627, 0.0014874
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017248, 0.0016960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012611, upper bound: 0.0012358
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012594, upper bound: 0.0012415
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025360, 0.0026074
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006319, 0.0006497
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034431, 0.0033488
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015242, 0.0015671
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006664, 0.0006482
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043305, 0.0042119
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010690, 0.0010991
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027659, 0.0028438
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014546, 0.0014955
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017341, 0.0016866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012288, upper bound: 0.0012629
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012261, upper bound: 0.0012670
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025277, 0.0026166
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006298, 0.0006520
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034552, 0.0033378
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015192, 0.0015726
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006687, 0.0006460
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043457, 0.0041981
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010655, 0.0011030
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027568, 0.0028538
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014498, 0.0015008
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017402, 0.0016811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012252, upper bound: 0.0012652
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012232, upper bound: 0.0012701
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025645, 0.0025783
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006390, 0.0006425
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034047, 0.0033864
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015413, 0.0015497
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006590, 0.0006554
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042822, 0.0042592
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010810, 0.0010869
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027970, 0.0028120
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014709, 0.0014788
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017148, 0.0017056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012740, upper bound: 0.0012317
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012728, upper bound: 0.0012352
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025561, 0.0025876
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006369, 0.0006448
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034168, 0.0033752
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015363, 0.0015552
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006613, 0.0006533
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042975, 0.0042452
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010775, 0.0010907
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027877, 0.0028221
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014660, 0.0014841
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017209, 0.0017000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012711, upper bound: 0.0012348
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012693, upper bound: 0.0012404
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025348, 0.0026082
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006316, 0.0006499
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034441, 0.0033471
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015235, 0.0015676
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006666, 0.0006478
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043318, 0.0042098
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010685, 0.0010995
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027645, 0.0028446
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014538, 0.0014960
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017346, 0.0016858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012276, upper bound: 0.0012728
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012256, upper bound: 0.0012771
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025256, 0.0026170
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006293, 0.0006521
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034557, 0.0033350
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015179, 0.0015729
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006688, 0.0006455
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043464, 0.0041945
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010646, 0.0011032
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027545, 0.0028542
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014486, 0.0015010
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017405, 0.0016797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012233, upper bound: 0.0012755
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012216, upper bound: 0.0012808
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025635, 0.0025796
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006388, 0.0006428
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034064, 0.0033851
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015408, 0.0015504
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006593, 0.0006552
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042843, 0.0042576
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010806, 0.0010874
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027959, 0.0028134
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014703, 0.0014796
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017156, 0.0017049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012637, upper bound: 0.0012327
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012618, upper bound: 0.0012365
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025546, 0.0025883
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006365, 0.0006449
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034178, 0.0033733
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015354, 0.0015556
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006615, 0.0006529
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042987, 0.0042428
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010769, 0.0010911
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027862, 0.0028229
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014652, 0.0014845
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017214, 0.0016990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0012350
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012595, upper bound: 0.0012415
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025405, 0.0026023
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006330, 0.0006484
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034363, 0.0033547
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015269, 0.0015640
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006651, 0.0006493
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043219, 0.0042193
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010709, 0.0010970
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027707, 0.0028382
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014571, 0.0014926
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017307, 0.0016896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012295, upper bound: 0.0012629
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012269, upper bound: 0.0012670
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025313, 0.0026112
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006307, 0.0006507
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034481, 0.0033425
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015214, 0.0015694
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006674, 0.0006469
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0043368, 0.0042040
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010670, 0.0011007
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027607, 0.0028479
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014518, 0.0014977
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017367, 0.0016835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012258, upper bound: 0.0012652
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012240, upper bound: 0.0012701
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025696, 0.0025745
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006403, 0.0006415
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0033996, 0.0033931
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015444, 0.0015473
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006580, 0.0006567
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042757, 0.0042677
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010832, 0.0010852
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0028025, 0.0028078
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014738, 0.0014766
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017122, 0.0017090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012756, upper bound: 0.0012317
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012748, upper bound: 0.0012352
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9894695, 0.9935201, 0.9894695, 0.9935201, -0.0025613, 0.0025830
1: -0.0038879, -0.0028786, -0.0038879, -0.0028786, -0.0006382, 0.0006436
2: 0.0052009, 0.0105496, 0.0052009, 0.0105496, -0.0034108, 0.0033822
3: -0.0060749, -0.0036403, -0.0060749, -0.0036403, -0.0015394, 0.0015524
4: 0.0015345, 0.0025697, 0.0015345, 0.0025697, -0.0006601, 0.0006546
5: 0.0055008, 0.0122281, 0.0055008, 0.0122281, -0.0042899, 0.0042539
6: -0.0015628, 0.0001447, -0.0015628, 0.0001447, -0.0010797, 0.0010888
7: -0.0071810, -0.0027633, -0.0071810, -0.0027633, -0.0027935, 0.0028171
8: -0.0033406, -0.0010173, -0.0033406, -0.0010173, -0.0014691, 0.0014815
9: -0.0006842, 0.0020097, -0.0006842, 0.0020097, -0.0017178, 0.0017034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012720, upper bound: 0.0012346
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012709, upper bound: 0.0012404
time: 0.90 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012404, upper bound: 0.0012709
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012346, upper bound: 0.0012720
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012352, upper bound: 0.0012748
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012317, upper bound: 0.0012756
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012701, upper bound: 0.0012240
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012652, upper bound: 0.0012258
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012670, upper bound: 0.0012269
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012629, upper bound: 0.0012295
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012415, upper bound: 0.0012595
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012350, upper bound: 0.0012612
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012365, upper bound: 0.0012618
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012327, upper bound: 0.0012637
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012808, upper bound: 0.0012215
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012755, upper bound: 0.0012233
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012771, upper bound: 0.0012256
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012728, upper bound: 0.0012276
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012404, upper bound: 0.0012693
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012348, upper bound: 0.0012711
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012352, upper bound: 0.0012728
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012318, upper bound: 0.0012740
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012701, upper bound: 0.0012233
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012652, upper bound: 0.0012252
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012670, upper bound: 0.0012261
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012629, upper bound: 0.0012288
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012415, upper bound: 0.0012594
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012358, upper bound: 0.0012611
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012366, upper bound: 0.0012616
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012636
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012817, upper bound: 0.0012215
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012771, upper bound: 0.0012233
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012775, upper bound: 0.0012250
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012740, upper bound: 0.0012273
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012304, upper bound: 0.0012720
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012267, upper bound: 0.0012737
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012267, upper bound: 0.0012752
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012234, upper bound: 0.0012771
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012668, upper bound: 0.0012318
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012624, upper bound: 0.0012338
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012634, upper bound: 0.0012339
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012603, upper bound: 0.0012380
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012324, upper bound: 0.0012620
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012276, upper bound: 0.0012649
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012285, upper bound: 0.0012641
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012251, upper bound: 0.0012676
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012797, upper bound: 0.0012304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012747, upper bound: 0.0012327
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012763, upper bound: 0.0012333
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012715, upper bound: 0.0012376
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012306, upper bound: 0.0012705
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012271, upper bound: 0.0012728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012267, upper bound: 0.0012739
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012234, upper bound: 0.0012759
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012668, upper bound: 0.0012311
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012625, upper bound: 0.0012338
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012634, upper bound: 0.0012337
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012603, upper bound: 0.0012379
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012324, upper bound: 0.0012620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012281, upper bound: 0.0012649
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012287, upper bound: 0.0012641
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012257, upper bound: 0.0012676
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012812, upper bound: 0.0012303
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012766, upper bound: 0.0012327
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012765, upper bound: 0.0012332
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012733, upper bound: 0.0012376
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012376, upper bound: 0.0012733
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012332, upper bound: 0.0012765
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012327, upper bound: 0.0012766
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012303, upper bound: 0.0012812
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012676, upper bound: 0.0012257
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012641, upper bound: 0.0012287
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012649, upper bound: 0.0012281
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012620, upper bound: 0.0012324
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012379, upper bound: 0.0012603
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012337, upper bound: 0.0012635
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012338, upper bound: 0.0012625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012311, upper bound: 0.0012668
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012759, upper bound: 0.0012234
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012739, upper bound: 0.0012267
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012728, upper bound: 0.0012271
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012705, upper bound: 0.0012305
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012376, upper bound: 0.0012715
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012333, upper bound: 0.0012763
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012327, upper bound: 0.0012747
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012304, upper bound: 0.0012797
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012676, upper bound: 0.0012251
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012641, upper bound: 0.0012285
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012649, upper bound: 0.0012276
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012620, upper bound: 0.0012324
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012380, upper bound: 0.0012603
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012339, upper bound: 0.0012634
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012338, upper bound: 0.0012625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012318, upper bound: 0.0012668
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012771, upper bound: 0.0012234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012752, upper bound: 0.0012267
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012737, upper bound: 0.0012267
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012720, upper bound: 0.0012304
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012273, upper bound: 0.0012740
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012250, upper bound: 0.0012775
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012233, upper bound: 0.0012771
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012215, upper bound: 0.0012817
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012636, upper bound: 0.0012334
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012616, upper bound: 0.0012366
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012611, upper bound: 0.0012358
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012594, upper bound: 0.0012415
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012288, upper bound: 0.0012629
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012261, upper bound: 0.0012670
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012252, upper bound: 0.0012652
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012232, upper bound: 0.0012701
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012740, upper bound: 0.0012317
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012728, upper bound: 0.0012352
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012711, upper bound: 0.0012348
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012693, upper bound: 0.0012404
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012276, upper bound: 0.0012728
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012256, upper bound: 0.0012771
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012233, upper bound: 0.0012755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012216, upper bound: 0.0012808
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012637, upper bound: 0.0012327
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012618, upper bound: 0.0012365
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0012350
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012595, upper bound: 0.0012415
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012295, upper bound: 0.0012629
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012269, upper bound: 0.0012670
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012258, upper bound: 0.0012652
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012240, upper bound: 0.0012701
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012756, upper bound: 0.0012317
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012748, upper bound: 0.0012352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012720, upper bound: 0.0012346
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0012709, upper bound: 0.0012404

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.08 + 415.31 = 418.39 seconds
