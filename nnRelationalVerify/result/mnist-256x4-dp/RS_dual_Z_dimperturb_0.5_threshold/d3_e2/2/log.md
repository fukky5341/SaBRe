## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00076797


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041546, -0.0041202, -0.0041546, -0.0041202, -0.0000295, 0.0000295)
1: (-0.0082449, -0.0069587, -0.0082449, -0.0069587, -0.0011044, 0.0011044)
2: (0.9665693, 0.9681127, 0.9665693, 0.9681127, -0.0013253, 0.0013253)
3: (-0.0002739, 0.0111109, -0.0002739, 0.0111109, -0.0097754, 0.0097754)
4: (-0.0015381, -0.0006722, -0.0015381, -0.0006722, -0.0007435, 0.0007435)
5: (0.0157158, 0.0165910, 0.0157158, 0.0165910, -0.0007514, 0.0007514)
6: (0.0038369, 0.0042626, 0.0038369, 0.0042626, -0.0003655, 0.0003655)
7: (-0.0106577, -0.0077073, -0.0106577, -0.0077073, -0.0025334, 0.0025334)
8: (0.0082738, 0.0106146, 0.0082738, 0.0106146, -0.0020099, 0.0020099)
9: (0.0126058, 0.0168159, 0.0126058, 0.0168159, -0.0036149, 0.0036149)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.60 + 1.59 = 3.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0008779, upper bound: 0.0008779

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008446, upper bound: 0.0008771
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008771, upper bound: 0.0008446
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 2, lower bound: -0.0008446, upper bound: 0.0008771
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 2, lower bound: -0.0008771, upper bound: 0.0008446

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041546, -0.0041202, -0.0041546, -0.0041202, -0.0000287, 0.0000290
1: -0.0082449, -0.0069587, -0.0082449, -0.0069587, -0.0010765, 0.0010854
2: 0.9665693, 0.9681127, 0.9665693, 0.9681127, -0.0012918, 0.0013026
3: -0.0002739, 0.0111109, -0.0002739, 0.0111109, -0.0095285, 0.0096075
4: -0.0015381, -0.0006722, -0.0015381, -0.0006722, -0.0007307, 0.0007247
5: 0.0157158, 0.0165910, 0.0157158, 0.0165910, -0.0007385, 0.0007324
6: 0.0038369, 0.0042626, 0.0038369, 0.0042626, -0.0003563, 0.0003592
7: -0.0106577, -0.0077073, -0.0106577, -0.0077073, -0.0024899, 0.0024694
8: 0.0082738, 0.0106146, 0.0082738, 0.0106146, -0.0019753, 0.0019591
9: 0.0126058, 0.0168159, 0.0126058, 0.0168159, -0.0035529, 0.0035236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007869, upper bound: 0.0007906
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007652, upper bound: 0.0008223
time: 0.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041546, -0.0041202, -0.0041546, -0.0041202, -0.0000290, 0.0000287
1: -0.0082449, -0.0069587, -0.0082449, -0.0069587, -0.0010854, 0.0010765
2: 0.9665693, 0.9681127, 0.9665693, 0.9681127, -0.0013026, 0.0012918
3: -0.0002739, 0.0111109, -0.0002739, 0.0111109, -0.0096075, 0.0095285
4: -0.0015381, -0.0006722, -0.0015381, -0.0006722, -0.0007247, 0.0007307
5: 0.0157158, 0.0165910, 0.0157158, 0.0165910, -0.0007324, 0.0007385
6: 0.0038369, 0.0042626, 0.0038369, 0.0042626, -0.0003592, 0.0003563
7: -0.0106577, -0.0077073, -0.0106577, -0.0077073, -0.0024694, 0.0024899
8: 0.0082738, 0.0106146, 0.0082738, 0.0106146, -0.0019591, 0.0019753
9: 0.0126058, 0.0168159, 0.0126058, 0.0168159, -0.0035236, 0.0035529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008223, upper bound: 0.0007652
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007906, upper bound: 0.0007869
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 2, lower bound: -0.0007869, upper bound: 0.0007906
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 2, lower bound: -0.0007652, upper bound: 0.0008223
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 2, lower bound: -0.0008223, upper bound: 0.0007652
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 2, lower bound: -0.0007906, upper bound: 0.0007869

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041546, -0.0041202, -0.0041546, -0.0041202, -0.0000246, 0.0000241
1: -0.0082449, -0.0069587, -0.0082449, -0.0069587, -0.0009205, 0.0009023
2: 0.9665693, 0.9681127, 0.9665693, 0.9681127, -0.0011047, 0.0010828
3: -0.0002739, 0.0111109, -0.0002739, 0.0111109, -0.0081480, 0.0079865
4: -0.0015381, -0.0006722, -0.0015381, -0.0006722, -0.0006074, 0.0006197
5: 0.0157158, 0.0165910, 0.0157158, 0.0165910, -0.0006139, 0.0006263
6: 0.0038369, 0.0042626, 0.0038369, 0.0042626, -0.0003046, 0.0002986
7: -0.0106577, -0.0077073, -0.0106577, -0.0077073, -0.0020698, 0.0021116
8: 0.0082738, 0.0106146, 0.0082738, 0.0106146, -0.0016421, 0.0016753
9: 0.0126058, 0.0168159, 0.0126058, 0.0168159, -0.0029534, 0.0030131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007454, upper bound: 0.0007658
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007624, upper bound: 0.0007407
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041546, -0.0041202, -0.0041546, -0.0041202, -0.0000239, 0.0000249
1: -0.0082449, -0.0069587, -0.0082449, -0.0069587, -0.0008934, 0.0009322
2: 0.9665693, 0.9681127, 0.9665693, 0.9681127, -0.0010721, 0.0011187
3: -0.0002739, 0.0111109, -0.0002739, 0.0111109, -0.0079074, 0.0082510
4: -0.0015381, -0.0006722, -0.0015381, -0.0006722, -0.0006275, 0.0006014
5: 0.0157158, 0.0165910, 0.0157158, 0.0165910, -0.0006342, 0.0006078
6: 0.0038369, 0.0042626, 0.0038369, 0.0042626, -0.0002956, 0.0003085
7: -0.0106577, -0.0077073, -0.0106577, -0.0077073, -0.0021383, 0.0020493
8: 0.0082738, 0.0106146, 0.0082738, 0.0106146, -0.0016964, 0.0016258
9: 0.0126058, 0.0168159, 0.0126058, 0.0168159, -0.0030512, 0.0029242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007232, upper bound: 0.0007995
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007394, upper bound: 0.0007729
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041546, -0.0041202, -0.0041546, -0.0041202, -0.0000249, 0.0000239
1: -0.0082449, -0.0069587, -0.0082449, -0.0069587, -0.0009322, 0.0008934
2: 0.9665693, 0.9681127, 0.9665693, 0.9681127, -0.0011187, 0.0010721
3: -0.0002739, 0.0111109, -0.0002739, 0.0111109, -0.0082510, 0.0079074
4: -0.0015381, -0.0006722, -0.0015381, -0.0006722, -0.0006014, 0.0006275
5: 0.0157158, 0.0165910, 0.0157158, 0.0165910, -0.0006078, 0.0006342
6: 0.0038369, 0.0042626, 0.0038369, 0.0042626, -0.0003085, 0.0002956
7: -0.0106577, -0.0077073, -0.0106577, -0.0077073, -0.0020493, 0.0021383
8: 0.0082738, 0.0106146, 0.0082738, 0.0106146, -0.0016258, 0.0016964
9: 0.0126058, 0.0168159, 0.0126058, 0.0168159, -0.0029242, 0.0030512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007729, upper bound: 0.0007394
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007994, upper bound: 0.0007232
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041546, -0.0041202, -0.0041546, -0.0041202, -0.0000241, 0.0000246
1: -0.0082449, -0.0069587, -0.0082449, -0.0069587, -0.0009023, 0.0009205
2: 0.9665693, 0.9681127, 0.9665693, 0.9681127, -0.0010828, 0.0011047
3: -0.0002739, 0.0111109, -0.0002739, 0.0111109, -0.0079865, 0.0081480
4: -0.0015381, -0.0006722, -0.0015381, -0.0006722, -0.0006197, 0.0006074
5: 0.0157158, 0.0165910, 0.0157158, 0.0165910, -0.0006263, 0.0006139
6: 0.0038369, 0.0042626, 0.0038369, 0.0042626, -0.0002986, 0.0003046
7: -0.0106577, -0.0077073, -0.0106577, -0.0077073, -0.0021116, 0.0020698
8: 0.0082738, 0.0106146, 0.0082738, 0.0106146, -0.0016753, 0.0016421
9: 0.0126058, 0.0168159, 0.0126058, 0.0168159, -0.0030131, 0.0029534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007406, upper bound: 0.0007624
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007658, upper bound: 0.0007454
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.33
Output dim: 2, lower bound: -0.0007454, upper bound: 0.0007658
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.33
Output dim: 2, lower bound: -0.0007624, upper bound: 0.0007407
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 2, lower bound: -0.0007232, upper bound: 0.0007995
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 2, lower bound: -0.0007394, upper bound: 0.0007729
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 2, lower bound: -0.0007729, upper bound: 0.0007394
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 2, lower bound: -0.0007994, upper bound: 0.0007232
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.33
Output dim: 2, lower bound: -0.0007406, upper bound: 0.0007624
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.33
Output dim: 2, lower bound: -0.0007658, upper bound: 0.0007454

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041546, -0.0041202, -0.0041546, -0.0041202, -0.0000235, 0.0000247
1: -0.0082449, -0.0069587, -0.0082449, -0.0069587, -0.0008800, 0.0009248
2: 0.9665693, 0.9681127, 0.9665693, 0.9681127, -0.0010560, 0.0011098
3: -0.0002739, 0.0111109, -0.0002739, 0.0111109, -0.0077889, 0.0081854
4: -0.0015381, -0.0006722, -0.0015381, -0.0006722, -0.0006225, 0.0005924
5: 0.0157158, 0.0165910, 0.0157158, 0.0165910, -0.0006292, 0.0005987
6: 0.0038369, 0.0042626, 0.0038369, 0.0042626, -0.0002912, 0.0003060
7: -0.0106577, -0.0077073, -0.0106577, -0.0077073, -0.0021213, 0.0020186
8: 0.0082738, 0.0106146, 0.0082738, 0.0106146, -0.0016830, 0.0016014
9: 0.0126058, 0.0168159, 0.0126058, 0.0168159, -0.0030270, 0.0028803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006945, upper bound: 0.0007655
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006945, upper bound: 0.0007655
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041546, -0.0041202, -0.0041546, -0.0041202, -0.0000239, 0.0000245
1: -0.0082449, -0.0069587, -0.0082449, -0.0069587, -0.0008934, 0.0009188
2: 0.9665693, 0.9681127, 0.9665693, 0.9681127, -0.0010721, 0.0011026
3: -0.0002739, 0.0111109, -0.0002739, 0.0111109, -0.0079074, 0.0081325
4: -0.0015381, -0.0006722, -0.0015381, -0.0006722, -0.0006185, 0.0006014
5: 0.0157158, 0.0165910, 0.0157158, 0.0165910, -0.0006251, 0.0006078
6: 0.0038369, 0.0042626, 0.0038369, 0.0042626, -0.0002956, 0.0003041
7: -0.0106577, -0.0077073, -0.0106577, -0.0077073, -0.0021076, 0.0020493
8: 0.0082738, 0.0106146, 0.0082738, 0.0106146, -0.0016721, 0.0016258
9: 0.0126058, 0.0168159, 0.0126058, 0.0168159, -0.0030074, 0.0029242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007088, upper bound: 0.0007428
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007088, upper bound: 0.0007429
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041546, -0.0041202, -0.0041546, -0.0041202, -0.0000245, 0.0000236
1: -0.0082449, -0.0069587, -0.0082449, -0.0069587, -0.0009188, 0.0008846
2: 0.9665693, 0.9681127, 0.9665693, 0.9681127, -0.0011026, 0.0010616
3: -0.0002739, 0.0111109, -0.0002739, 0.0111109, -0.0081325, 0.0078303
4: -0.0015381, -0.0006722, -0.0015381, -0.0006722, -0.0005955, 0.0006185
5: 0.0157158, 0.0165910, 0.0157158, 0.0165910, -0.0006019, 0.0006251
6: 0.0038369, 0.0042626, 0.0038369, 0.0042626, -0.0003041, 0.0002928
7: -0.0106577, -0.0077073, -0.0106577, -0.0077073, -0.0020293, 0.0021076
8: 0.0082738, 0.0106146, 0.0082738, 0.0106146, -0.0016099, 0.0016721
9: 0.0126058, 0.0168159, 0.0126058, 0.0168159, -0.0028956, 0.0030074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007429, upper bound: 0.0007088
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007429, upper bound: 0.0007088
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041546, -0.0041202, -0.0041546, -0.0041202, -0.0000249, 0.0000235
1: -0.0082449, -0.0069587, -0.0082449, -0.0069587, -0.0009322, 0.0008800
2: 0.9665693, 0.9681127, 0.9665693, 0.9681127, -0.0011187, 0.0010560
3: -0.0002739, 0.0111109, -0.0002739, 0.0111109, -0.0082510, 0.0077889
4: -0.0015381, -0.0006722, -0.0015381, -0.0006722, -0.0005924, 0.0006275
5: 0.0157158, 0.0165910, 0.0157158, 0.0165910, -0.0005987, 0.0006342
6: 0.0038369, 0.0042626, 0.0038369, 0.0042626, -0.0003085, 0.0002912
7: -0.0106577, -0.0077073, -0.0106577, -0.0077073, -0.0020186, 0.0021383
8: 0.0082738, 0.0106146, 0.0082738, 0.0106146, -0.0016014, 0.0016964
9: 0.0126058, 0.0168159, 0.0126058, 0.0168159, -0.0028803, 0.0030512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007656, upper bound: 0.0006945
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007655, upper bound: 0.0006944
time: 0.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.09
Output dim: 2, lower bound: -0.0006945, upper bound: 0.0007655
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.09
Output dim: 2, lower bound: -0.0006945, upper bound: 0.0007655
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.09
Output dim: 2, lower bound: -0.0007088, upper bound: 0.0007428
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.09
Output dim: 2, lower bound: -0.0007088, upper bound: 0.0007429
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.09
Output dim: 2, lower bound: -0.0007429, upper bound: 0.0007088
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.09
Output dim: 2, lower bound: -0.0007429, upper bound: 0.0007088
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.09
Output dim: 2, lower bound: -0.0007656, upper bound: 0.0006945
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.09
Output dim: 2, lower bound: -0.0007655, upper bound: 0.0006944

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.19 + 33.74 = 36.93 seconds
