## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0016185600000000002


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013854, 0.0013854)
1: (-0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035157, 0.0035157)
2: (0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021812, 0.0021812)
3: (0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040728, 0.0040728)
4: (-0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035761, 0.0035761)
5: (0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013545, 0.0013545)
6: (0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051689, 0.0051689)
7: (0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036170, 0.0036170)
8: (-0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038780, 0.0038780)
9: (-0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025616, 0.0025616)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 2.80 = 4.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0020232, upper bound: 0.0020232

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0020060, upper bound: 0.0020055
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0020055, upper bound: 0.0020060
time: 1.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.57
Output dim: 7, lower bound: -0.0020060, upper bound: 0.0020055
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.57
Output dim: 7, lower bound: -0.0020055, upper bound: 0.0020060

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013718, 0.0013727
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0034811, 0.0034834
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021597, 0.0021611
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040354, 0.0040327
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035409, 0.0035432
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013412, 0.0013421
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051214, 0.0051180
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0035837, 0.0035814
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038423, 0.0038398
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025364, 0.0025381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018912, upper bound: 0.0018912
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018912, upper bound: 0.0018912
time: 1.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013727, 0.0013718
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0034834, 0.0034811
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021611, 0.0021597
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040327, 0.0040354
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035432, 0.0035409
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013421, 0.0013412
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051180, 0.0051214
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0035814, 0.0035837
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038398, 0.0038423
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025381, 0.0025364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019104, upper bound: 0.0019104
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019104, upper bound: 0.0019104
time: 1.24 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 7, lower bound: -0.0018912, upper bound: 0.0018912
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 7, lower bound: -0.0018912, upper bound: 0.0018912
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 7, lower bound: -0.0019104, upper bound: 0.0019104
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 7, lower bound: -0.0019104, upper bound: 0.0019104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013607, 0.0013635
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0034529, 0.0034601
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021422, 0.0021467
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040084, 0.0040000
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035121, 0.0035195
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013303, 0.0013331
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0050872, 0.0050765
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0035598, 0.0035523
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038166, 0.0038086
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025158, 0.0025211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018238, upper bound: 0.0018238
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018238, upper bound: 0.0018238
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013626, 0.0013727
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0034578, 0.0034834
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021453, 0.0021611
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040354, 0.0040057
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035172, 0.0035432
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013322, 0.0013421
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051214, 0.0050838
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0035837, 0.0035574
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038423, 0.0038141
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025194, 0.0025381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017218, upper bound: 0.0017103
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017103, upper bound: 0.0017218
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013539, 0.0013577
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0034357, 0.0034453
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021315, 0.0021375
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039913, 0.0039801
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034947, 0.0035045
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013237, 0.0013274
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0050654, 0.0050512
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0035445, 0.0035346
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038003, 0.0037897
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025033, 0.0025103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018918, upper bound: 0.0018917
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018917, upper bound: 0.0018918
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013586, 0.0013718
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0034476, 0.0034811
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021389, 0.0021597
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040327, 0.0039939
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035068, 0.0035409
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013283, 0.0013412
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051180, 0.0050688
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0035814, 0.0035469
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038398, 0.0038028
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025120, 0.0025364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018918, upper bound: 0.0018917
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018917, upper bound: 0.0018918
time: 1.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.50 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 7, lower bound: -0.0018238, upper bound: 0.0018238
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 7, lower bound: -0.0018238, upper bound: 0.0018238
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 7, lower bound: -0.0017218, upper bound: 0.0017103
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 7, lower bound: -0.0017103, upper bound: 0.0017218
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 7, lower bound: -0.0018918, upper bound: 0.0018917
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 7, lower bound: -0.0018917, upper bound: 0.0018918
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 7, lower bound: -0.0018918, upper bound: 0.0018917
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 7, lower bound: -0.0018917, upper bound: 0.0018918

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013295, 0.0013333
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033738, 0.0033836
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020931, 0.0020992
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039197, 0.0039084
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034317, 0.0034416
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012999, 0.0013036
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049746, 0.0049603
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034810, 0.0034710
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037322, 0.0037214
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024582, 0.0024653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017893, upper bound: 0.0018033
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018034, upper bound: 0.0017893
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013305, 0.0013324
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033763, 0.0033813
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020947, 0.0020978
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039170, 0.0039113
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034342, 0.0034393
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013008, 0.0013027
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049712, 0.0049639
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034786, 0.0034735
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037296, 0.0037241
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024600, 0.0024636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011933, upper bound: 0.0011933
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011933, upper bound: 0.0011933
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013279, 0.0013585
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033697, 0.0034474
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020906, 0.0021388
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039936, 0.0039037
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034276, 0.0035066
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012983, 0.0013282
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0050684, 0.0049543
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0035466, 0.0034668
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038026, 0.0037169
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024552, 0.0025118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016676, upper bound: 0.0016234
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016346, upper bound: 0.0016572
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013626, 0.0013380
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0034578, 0.0033953
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021453, 0.0021065
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039333, 0.0040057
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035172, 0.0034536
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013322, 0.0013081
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049919, 0.0050838
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034931, 0.0035574
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037451, 0.0038141
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025194, 0.0024739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016832, upper bound: 0.0016950
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016830, upper bound: 0.0016950
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013514, 0.0013556
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0034293, 0.0034401
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021276, 0.0021342
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039852, 0.0039727
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034882, 0.0034991
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013212, 0.0013254
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0050577, 0.0050419
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0035391, 0.0035281
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037945, 0.0037826
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024986, 0.0025065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018642, upper bound: 0.0018641
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018642, upper bound: 0.0018642
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013518, 0.0013552
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0034305, 0.0034390
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021283, 0.0021336
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039839, 0.0039740
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034894, 0.0034980
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013217, 0.0013250
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0050561, 0.0050436
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0035380, 0.0035293
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037933, 0.0037839
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024995, 0.0025057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018743, upper bound: 0.0018613
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018612, upper bound: 0.0018744
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013561, 0.0013697
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0034413, 0.0034757
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021350, 0.0021563
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040264, 0.0039866
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035004, 0.0035354
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013258, 0.0013391
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051101, 0.0050595
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0035758, 0.0035404
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038338, 0.0037958
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025074, 0.0025324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018469, upper bound: 0.0018171
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018174, upper bound: 0.0018468
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013565, 0.0013692
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0034424, 0.0034746
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021357, 0.0021557
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040252, 0.0039879
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035015, 0.0035343
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013263, 0.0013387
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051085, 0.0050611
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0035747, 0.0035415
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038326, 0.0037971
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025082, 0.0025316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018568, upper bound: 0.0018718
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018715, upper bound: 0.0018573
time: 1.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0017893, upper bound: 0.0018033
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0018034, upper bound: 0.0017893
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0011933, upper bound: 0.0011933
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0011933, upper bound: 0.0011933
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0016676, upper bound: 0.0016234
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0016346, upper bound: 0.0016572
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0016832, upper bound: 0.0016950
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0016830, upper bound: 0.0016950
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0018642, upper bound: 0.0018641
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0018642, upper bound: 0.0018642
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0018743, upper bound: 0.0018613
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0018612, upper bound: 0.0018744
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0018469, upper bound: 0.0018171
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0018174, upper bound: 0.0018468
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0018568, upper bound: 0.0018718
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.38
Output dim: 7, lower bound: -0.0018715, upper bound: 0.0018573

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012953, 0.0012953
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032870, 0.0032870
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020393, 0.0020393
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038078, 0.0038078
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033434, 0.0033434
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012664, 0.0012664
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048326, 0.0048326
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033816, 0.0033816
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036256, 0.0036256
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023949, 0.0023949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017454, upper bound: 0.0017055
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016943, upper bound: 0.0017599
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012915, 0.0012983
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032773, 0.0032947
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020332, 0.0020440
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038167, 0.0037966
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033335, 0.0033512
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012626, 0.0012693
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048439, 0.0048183
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033895, 0.0033716
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036341, 0.0036149
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023879, 0.0024005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018029, upper bound: 0.0017888
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018027, upper bound: 0.0017888
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012515, 0.0012911
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031757, 0.0032763
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019702, 0.0020326
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037954, 0.0036789
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032303, 0.0033326
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012235, 0.0012623
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048169, 0.0046690
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033706, 0.0032672
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036139, 0.0035029
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023139, 0.0023872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016646, upper bound: 0.0016194
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016645, upper bound: 0.0016194
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012614, 0.0012821
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032009, 0.0032534
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019859, 0.0020184
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037689, 0.0037081
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032559, 0.0033093
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012332, 0.0012535
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047832, 0.0047061
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033471, 0.0032931
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035886, 0.0035307
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023322, 0.0023705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015895, upper bound: 0.0016150
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015915, upper bound: 0.0016135
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012653, 0.0012491
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032109, 0.0031698
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019920, 0.0019665
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036720, 0.0037196
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032660, 0.0032242
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012371, 0.0012212
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046603, 0.0047207
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032610, 0.0033033
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034963, 0.0035417
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023395, 0.0023095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015902, upper bound: 0.0015939
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015902, upper bound: 0.0015939
time: 1.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012781, 0.0012363
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032435, 0.0031374
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020123, 0.0019464
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036345, 0.0037574
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032992, 0.0031912
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012496, 0.0012087
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046126, 0.0047686
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032277, 0.0033369
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034606, 0.0035776
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023632, 0.0022859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016826, upper bound: 0.0016945
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016827, upper bound: 0.0016946
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012534, 0.0012681
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031807, 0.0032180
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019733, 0.0019965
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037279, 0.0036847
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032353, 0.0032733
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012254, 0.0012398
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047312, 0.0046764
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033107, 0.0032723
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035496, 0.0035084
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023175, 0.0023447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018634, upper bound: 0.0018631
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018632, upper bound: 0.0018632
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012639, 0.0012549
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032073, 0.0031844
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019898, 0.0019756
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036889, 0.0037155
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032623, 0.0032390
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012357, 0.0012269
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046817, 0.0047154
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032761, 0.0032996
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035125, 0.0035377
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023369, 0.0023202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0013161, upper bound: 0.0013156
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0013161, upper bound: 0.0013156
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013273, 0.0013418
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033683, 0.0034050
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020897, 0.0021125
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039446, 0.0039020
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034261, 0.0034635
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012977, 0.0013119
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0050062, 0.0049522
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0035031, 0.0034653
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037558, 0.0037153
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024542, 0.0024809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018302, upper bound: 0.0018287
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018417, upper bound: 0.0018143
time: 1.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013379, 0.0013307
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033950, 0.0033768
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021063, 0.0020950
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039119, 0.0039330
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034533, 0.0034348
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013080, 0.0013010
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049647, 0.0049914
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034740, 0.0034928
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037247, 0.0037448
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024737, 0.0024604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018134, upper bound: 0.0017982
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017876, upper bound: 0.0018295
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012784, 0.0013028
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032441, 0.0033062
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020127, 0.0020512
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038300, 0.0037582
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032998, 0.0033629
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012499, 0.0012738
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048608, 0.0047696
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034014, 0.0033376
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036468, 0.0035784
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023637, 0.0024089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017832, upper bound: 0.0017547
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017832, upper bound: 0.0017541
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012891, 0.0012930
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032714, 0.0032812
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020296, 0.0020357
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038012, 0.0037897
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033275, 0.0033376
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012604, 0.0012642
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048242, 0.0048097
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033757, 0.0033656
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036193, 0.0036084
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023836, 0.0023907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017543, upper bound: 0.0017832
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017550, upper bound: 0.0017832
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013217, 0.0013315
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033539, 0.0033788
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020808, 0.0020962
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039142, 0.0038853
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034115, 0.0034368
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012922, 0.0013018
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049676, 0.0049310
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034761, 0.0034505
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037269, 0.0036995
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024437, 0.0024618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018092, upper bound: 0.0017568
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017416, upper bound: 0.0018256
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013187, 0.0013349
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033464, 0.0033875
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020761, 0.0021016
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039242, 0.0038767
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034039, 0.0034456
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012893, 0.0013051
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049804, 0.0049200
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034850, 0.0034428
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037365, 0.0036912
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024382, 0.0024682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018265, upper bound: 0.0017840
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017970, upper bound: 0.0018127
time: 1.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0017454, upper bound: 0.0017055
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0016943, upper bound: 0.0017599
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0018029, upper bound: 0.0017888
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0018027, upper bound: 0.0017888
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0016646, upper bound: 0.0016194
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0016645, upper bound: 0.0016194
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0015895, upper bound: 0.0016150
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0015915, upper bound: 0.0016135
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0015902, upper bound: 0.0015939
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0015902, upper bound: 0.0015939
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0016826, upper bound: 0.0016945
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0016827, upper bound: 0.0016946
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0018634, upper bound: 0.0018631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0018632, upper bound: 0.0018632
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0013161, upper bound: 0.0013156
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0013161, upper bound: 0.0013156
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0018302, upper bound: 0.0018287
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0018417, upper bound: 0.0018143
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0018134, upper bound: 0.0017982
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0017876, upper bound: 0.0018295
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0017832, upper bound: 0.0017547
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0017832, upper bound: 0.0017541
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0017543, upper bound: 0.0017832
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0017550, upper bound: 0.0017832
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0018092, upper bound: 0.0017568
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0017416, upper bound: 0.0018256
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0018265, upper bound: 0.0017840
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.0017970, upper bound: 0.0018127

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012315, 0.0012473
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031251, 0.0031652
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019388, 0.0019637
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036667, 0.0036203
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031787, 0.0032195
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012040, 0.0012195
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046535, 0.0045946
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032563, 0.0032151
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034913, 0.0034471
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022770, 0.0023062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015404, upper bound: 0.0015001
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015368, upper bound: 0.0015043
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012466, 0.0012315
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031635, 0.0031251
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019627, 0.0019388
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036203, 0.0036648
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032179, 0.0031787
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012188, 0.0012040
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045946, 0.0046511
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032151, 0.0032546
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034471, 0.0034895
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023050, 0.0022770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016679, upper bound: 0.0017331
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016684, upper bound: 0.0017332
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012918, 0.0012988
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032780, 0.0032958
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020337, 0.0020447
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038180, 0.0037974
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033343, 0.0033524
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012629, 0.0012698
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048456, 0.0048194
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033907, 0.0033724
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036353, 0.0036157
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023884, 0.0024014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011495, upper bound: 0.0011487
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011495, upper bound: 0.0011487
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012918, 0.0012986
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032782, 0.0032954
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020338, 0.0020445
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038176, 0.0037977
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033345, 0.0033520
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012630, 0.0012696
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048450, 0.0048198
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033903, 0.0033726
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036349, 0.0036160
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023886, 0.0024011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017845, upper bound: 0.0017511
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017673, upper bound: 0.0017709
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012481, 0.0012882
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031671, 0.0032691
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019649, 0.0020282
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037871, 0.0036690
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032215, 0.0033252
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012202, 0.0012595
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048063, 0.0046564
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033632, 0.0032583
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036059, 0.0034935
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023076, 0.0023819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015647, upper bound: 0.0015255
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015647, upper bound: 0.0015255
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012485, 0.0012877
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031682, 0.0032677
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019656, 0.0020273
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037855, 0.0036702
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032226, 0.0033238
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012206, 0.0012590
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048043, 0.0046580
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033618, 0.0032595
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036044, 0.0034946
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023084, 0.0023809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016641, upper bound: 0.0016183
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016639, upper bound: 0.0016169
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012774, 0.0012357
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032416, 0.0031358
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020111, 0.0019455
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036327, 0.0037552
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032972, 0.0031897
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012489, 0.0012082
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046104, 0.0047659
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032261, 0.0033349
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034589, 0.0035756
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023619, 0.0022848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016750, upper bound: 0.0016888
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016748, upper bound: 0.0016892
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012776, 0.0012358
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032420, 0.0031359
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020113, 0.0019455
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036328, 0.0037557
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032976, 0.0031897
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012490, 0.0012082
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046105, 0.0047664
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032262, 0.0033353
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034590, 0.0035760
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023621, 0.0022849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016295, upper bound: 0.0016068
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015959, upper bound: 0.0016402
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012526, 0.0012675
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031788, 0.0032164
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019721, 0.0019955
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037261, 0.0036824
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032333, 0.0032716
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012247, 0.0012392
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047289, 0.0046735
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033090, 0.0032703
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035478, 0.0035063
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023161, 0.0023435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018288, upper bound: 0.0018433
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018434, upper bound: 0.0018281
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012526, 0.0012673
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031787, 0.0032161
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019721, 0.0019953
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037257, 0.0036824
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032333, 0.0032713
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012247, 0.0012391
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047283, 0.0046734
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033087, 0.0032702
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035474, 0.0035062
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023161, 0.0023433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018288, upper bound: 0.0018433
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018433, upper bound: 0.0018282
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012699, 0.0012844
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032226, 0.0032594
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019993, 0.0020222
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037759, 0.0037332
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032779, 0.0033154
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012416, 0.0012558
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047921, 0.0047379
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033533, 0.0033154
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035952, 0.0035546
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023480, 0.0023749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0012733, upper bound: 0.0012785
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0012733, upper bound: 0.0012785
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012700, 0.0012836
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032227, 0.0032572
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019994, 0.0020208
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037733, 0.0037333
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032780, 0.0033131
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012416, 0.0012549
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047888, 0.0047381
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033510, 0.0033155
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035928, 0.0035547
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023481, 0.0023732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0012802, upper bound: 0.0012722
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0012802, upper bound: 0.0012722
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012720, 0.0012747
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032278, 0.0032348
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020026, 0.0020069
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037474, 0.0037393
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032833, 0.0032903
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012436, 0.0012463
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047559, 0.0047457
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033279, 0.0033208
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035681, 0.0035604
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023518, 0.0023569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018122, upper bound: 0.0017970
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018111, upper bound: 0.0017966
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012833, 0.0012648
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032565, 0.0032096
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020203, 0.0019913
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037182, 0.0037725
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033124, 0.0032647
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012546, 0.0012366
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047189, 0.0047877
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033020, 0.0033502
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035403, 0.0035920
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023727, 0.0023386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015417, upper bound: 0.0015890
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015396, upper bound: 0.0015910
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012457, 0.0012708
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031611, 0.0032249
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019612, 0.0020007
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037359, 0.0036620
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032154, 0.0032802
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012179, 0.0012425
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047413, 0.0046475
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033177, 0.0032521
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035571, 0.0034868
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023032, 0.0023497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017360, upper bound: 0.0016392
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016699, upper bound: 0.0017092
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012466, 0.0012694
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031634, 0.0032214
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019626, 0.0019986
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037318, 0.0036646
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032177, 0.0032767
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012188, 0.0012411
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047362, 0.0046509
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033141, 0.0032545
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035533, 0.0034893
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023049, 0.0023471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017360, upper bound: 0.0016392
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016699, upper bound: 0.0017087
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012557, 0.0012610
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031865, 0.0032000
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019769, 0.0019853
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037070, 0.0036914
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032412, 0.0032549
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012277, 0.0012329
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047047, 0.0046848
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032921, 0.0032782
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035296, 0.0035148
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023217, 0.0023315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017266, upper bound: 0.0017556
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017266, upper bound: 0.0017556
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012573, 0.0012600
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031906, 0.0031975
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019795, 0.0019837
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037041, 0.0036962
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032454, 0.0032524
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012293, 0.0012319
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047010, 0.0046909
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032895, 0.0032825
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035269, 0.0035193
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023247, 0.0023297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017102, upper bound: 0.0016710
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016384, upper bound: 0.0017355
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012583, 0.0012834
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031930, 0.0032567
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019810, 0.0020205
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037728, 0.0036990
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032478, 0.0033126
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012302, 0.0012547
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047881, 0.0046945
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033505, 0.0032850
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035923, 0.0035220
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023265, 0.0023729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018088, upper bound: 0.0017558
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018087, upper bound: 0.0017564
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012742, 0.0012677
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032336, 0.0032169
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020061, 0.0019958
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037266, 0.0037459
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032891, 0.0032721
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012458, 0.0012394
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047295, 0.0047541
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033095, 0.0033267
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035483, 0.0035667
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023560, 0.0023439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017135, upper bound: 0.0017976
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017149, upper bound: 0.0017983
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012413, 0.0012690
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031500, 0.0032202
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019543, 0.0019978
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037304, 0.0036492
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032041, 0.0032755
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012136, 0.0012407
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047344, 0.0046312
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033129, 0.0032407
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035520, 0.0034746
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022951, 0.0023463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017988, upper bound: 0.0017555
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017988, upper bound: 0.0017558
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012519, 0.0012585
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031770, 0.0031936
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019710, 0.0019813
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036996, 0.0036804
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032315, 0.0032484
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012240, 0.0012304
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046953, 0.0046709
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032856, 0.0032684
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035226, 0.0035043
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023148, 0.0023269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017957, upper bound: 0.0018115
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017960, upper bound: 0.0018121
time: 1.74 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0015404, upper bound: 0.0015001
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0015368, upper bound: 0.0015043
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0016679, upper bound: 0.0017331
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0016684, upper bound: 0.0017332
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0011495, upper bound: 0.0011487
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0011495, upper bound: 0.0011487
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017845, upper bound: 0.0017511
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017673, upper bound: 0.0017709
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0015647, upper bound: 0.0015255
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0015647, upper bound: 0.0015255
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0016641, upper bound: 0.0016183
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0016639, upper bound: 0.0016169
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0016750, upper bound: 0.0016888
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0016748, upper bound: 0.0016892
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0016295, upper bound: 0.0016068
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0015959, upper bound: 0.0016402
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0018288, upper bound: 0.0018433
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0018434, upper bound: 0.0018281
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0018288, upper bound: 0.0018433
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0018433, upper bound: 0.0018282
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0012733, upper bound: 0.0012785
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0012733, upper bound: 0.0012785
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0012802, upper bound: 0.0012722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0012802, upper bound: 0.0012722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0018122, upper bound: 0.0017970
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0018111, upper bound: 0.0017966
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0015417, upper bound: 0.0015890
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0015396, upper bound: 0.0015910
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017360, upper bound: 0.0016392
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0016699, upper bound: 0.0017092
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017360, upper bound: 0.0016392
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0016699, upper bound: 0.0017087
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017266, upper bound: 0.0017556
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017266, upper bound: 0.0017556
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017102, upper bound: 0.0016710
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0016384, upper bound: 0.0017355
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0018088, upper bound: 0.0017558
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0018087, upper bound: 0.0017564
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017135, upper bound: 0.0017976
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017149, upper bound: 0.0017983
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017988, upper bound: 0.0017555
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017988, upper bound: 0.0017558
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017957, upper bound: 0.0018115
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.24
Output dim: 7, lower bound: -0.0017960, upper bound: 0.0018121

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011316, 0.0011277
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028715, 0.0028616
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017815, 0.0017753
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0033150, 0.0033265
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029208, 0.0029107
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011063, 0.0011025
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0042072, 0.0042217
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029440, 0.0029542
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031564, 0.0031673
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020922, 0.0020850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016179, upper bound: 0.0016523
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015851, upper bound: 0.0016778
time: 1.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011428, 0.0011149
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029000, 0.0028292
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017992, 0.0017552
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032775, 0.0033596
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029498, 0.0028777
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011173, 0.0010900
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041595, 0.0042637
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029106, 0.0029835
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031207, 0.0031988
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021130, 0.0020614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016583, upper bound: 0.0017270
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016584, upper bound: 0.0017274
time: 2.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012691, 0.0012876
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032204, 0.0032676
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019980, 0.0020272
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037853, 0.0037307
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032757, 0.0033237
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012407, 0.0012589
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048041, 0.0047347
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033617, 0.0033131
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036042, 0.0035522
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023464, 0.0023808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017575, upper bound: 0.0017239
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017575, upper bound: 0.0017239
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012813, 0.0012758
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032514, 0.0032376
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020172, 0.0020086
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037506, 0.0037666
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033072, 0.0032932
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012527, 0.0012474
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047600, 0.0047802
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033308, 0.0033450
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035711, 0.0035863
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023690, 0.0023589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0017439
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017406, upper bound: 0.0017439
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012502, 0.0012896
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031725, 0.0032725
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019682, 0.0020303
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037911, 0.0036751
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032269, 0.0033287
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012223, 0.0012608
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048114, 0.0046642
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033668, 0.0032638
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036097, 0.0034993
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023115, 0.0023844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015636, upper bound: 0.0015241
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015636, upper bound: 0.0015241
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012503, 0.0012893
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031728, 0.0032718
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019684, 0.0020298
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037902, 0.0036756
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032273, 0.0033280
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012224, 0.0012605
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048103, 0.0046648
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033660, 0.0032642
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036089, 0.0034997
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023118, 0.0023839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016371, upper bound: 0.0015899
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016368, upper bound: 0.0015899
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012734, 0.0012320
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032315, 0.0031263
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020049, 0.0019396
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036217, 0.0037436
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032870, 0.0031800
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012450, 0.0012045
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045963, 0.0047511
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032163, 0.0033246
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034484, 0.0035645
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023545, 0.0022779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016264, upper bound: 0.0015842
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015741, upper bound: 0.0016385
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012739, 0.0012319
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032326, 0.0031261
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020055, 0.0019394
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036214, 0.0037448
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032881, 0.0031798
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012454, 0.0012044
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045961, 0.0047527
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032161, 0.0033257
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034482, 0.0035657
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023553, 0.0022777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016394, upper bound: 0.0016684
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016540, upper bound: 0.0016535
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011907, 0.0011597
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030216, 0.0029428
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018746, 0.0018257
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034091, 0.0035003
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030734, 0.0029933
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011641, 0.0011338
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043266, 0.0044424
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030275, 0.0031086
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032460, 0.0033329
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022015, 0.0021441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015360, upper bound: 0.0015049
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015360, upper bound: 0.0015049
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012007, 0.0011487
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030470, 0.0029149
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018904, 0.0018084
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0033767, 0.0035298
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030993, 0.0029649
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011739, 0.0011230
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0042855, 0.0044798
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029988, 0.0031347
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032152, 0.0033609
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022201, 0.0021238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015616, upper bound: 0.0016194
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015753, upper bound: 0.0016034
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012237, 0.0012338
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031054, 0.0031308
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019266, 0.0019424
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036269, 0.0035975
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031587, 0.0031846
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011964, 0.0012062
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046030, 0.0045657
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032210, 0.0031948
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034534, 0.0034254
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022627, 0.0022812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015927, upper bound: 0.0016020
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015898, upper bound: 0.0016042
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012189, 0.0012371
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030932, 0.0031394
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019190, 0.0019477
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036369, 0.0035833
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031463, 0.0031933
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011917, 0.0012095
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046156, 0.0045476
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032298, 0.0031822
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034629, 0.0034118
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022537, 0.0022874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017980, upper bound: 0.0017543
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017683, upper bound: 0.0017830
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012237, 0.0012336
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031054, 0.0031305
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019266, 0.0019422
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036265, 0.0035974
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031587, 0.0031842
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011964, 0.0012061
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046025, 0.0045656
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032206, 0.0031948
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034530, 0.0034253
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022626, 0.0022809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017655, upper bound: 0.0017807
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017656, upper bound: 0.0017806
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012189, 0.0012370
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030931, 0.0031390
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019190, 0.0019475
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036364, 0.0035832
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031462, 0.0031929
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011917, 0.0012094
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046151, 0.0045476
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032294, 0.0031822
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034625, 0.0034118
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022537, 0.0022871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0012911, upper bound: 0.0012888
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0012911, upper bound: 0.0012888
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012734, 0.0012764
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032315, 0.0032391
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020049, 0.0020096
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037524, 0.0037436
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032870, 0.0032947
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012450, 0.0012480
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047623, 0.0047511
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033324, 0.0033246
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035729, 0.0035645
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023545, 0.0023601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017640, upper bound: 0.0017644
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017795, upper bound: 0.0017512
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012736, 0.0012762
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032320, 0.0032385
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020052, 0.0020092
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037516, 0.0037441
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032875, 0.0032941
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012452, 0.0012477
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047613, 0.0047518
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033317, 0.0033251
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035722, 0.0035650
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023549, 0.0023596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017824, upper bound: 0.0017676
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017827, upper bound: 0.0017676
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011793, 0.0012191
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029927, 0.0030937
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018567, 0.0019194
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035839, 0.0034669
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030441, 0.0031468
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011530, 0.0011919
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045485, 0.0044000
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031828, 0.0030789
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034125, 0.0033011
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021805, 0.0022541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017202, upper bound: 0.0016153
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017013, upper bound: 0.0016220
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011955, 0.0012043
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030339, 0.0030561
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018822, 0.0018960
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035403, 0.0035146
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030860, 0.0031085
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011689, 0.0011774
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044931, 0.0044605
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031441, 0.0031212
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033709, 0.0033464
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022105, 0.0022267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016366, upper bound: 0.0016884
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016490, upper bound: 0.0016741
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011802, 0.0012185
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029950, 0.0030921
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018581, 0.0019184
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035821, 0.0034696
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030464, 0.0031452
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011539, 0.0011913
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045461, 0.0044033
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031812, 0.0030812
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034107, 0.0033036
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021822, 0.0022530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016917, upper bound: 0.0016055
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017020, upper bound: 0.0015954
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011960, 0.0012029
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030351, 0.0030526
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018830, 0.0018938
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035363, 0.0035160
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030872, 0.0031050
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011693, 0.0011761
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044880, 0.0044622
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031405, 0.0031225
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033671, 0.0033478
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022114, 0.0022241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016542, upper bound: 0.0016790
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016445, upper bound: 0.0016907
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011445, 0.0011602
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029044, 0.0029441
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018019, 0.0018266
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034106, 0.0033646
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029542, 0.0029947
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011190, 0.0011343
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043286, 0.0042701
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030289, 0.0029880
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032475, 0.0032036
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021162, 0.0021451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017051, upper bound: 0.0017205
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016943, upper bound: 0.0017364
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011565, 0.0011469
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029348, 0.0029105
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018208, 0.0018057
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0033717, 0.0033998
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029852, 0.0029605
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011307, 0.0011213
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0042791, 0.0043148
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029943, 0.0030193
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032104, 0.0032372
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021383, 0.0021206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016921, upper bound: 0.0017355
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017065, upper bound: 0.0017210
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011910, 0.0012097
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030223, 0.0030699
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018750, 0.0019045
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035563, 0.0035011
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030741, 0.0031226
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011644, 0.0011827
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045134, 0.0044434
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031582, 0.0031093
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033861, 0.0033336
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022021, 0.0022367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016762, upper bound: 0.0016501
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016895, upper bound: 0.0016371
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012058, 0.0011935
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030598, 0.0030286
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018983, 0.0018790
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035085, 0.0035447
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031124, 0.0030806
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011789, 0.0011669
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044528, 0.0044987
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031158, 0.0031479
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033407, 0.0033751
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022294, 0.0022067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016055, upper bound: 0.0017149
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016175, upper bound: 0.0016992
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012582, 0.0012835
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031928, 0.0032571
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019808, 0.0020207
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037732, 0.0036987
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032476, 0.0033130
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012301, 0.0012549
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047886, 0.0046941
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033508, 0.0032847
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035926, 0.0035217
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023263, 0.0023731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017631, upper bound: 0.0016809
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017374, upper bound: 0.0017122
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012583, 0.0012833
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031932, 0.0032565
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019811, 0.0020204
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037726, 0.0036991
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032480, 0.0033125
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012303, 0.0012547
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047879, 0.0046947
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033503, 0.0032851
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035921, 0.0035222
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023266, 0.0023728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017647, upper bound: 0.0017232
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017741, upper bound: 0.0017127
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011550, 0.0011615
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029311, 0.0029474
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018184, 0.0018286
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034144, 0.0033955
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029814, 0.0029980
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011293, 0.0011356
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043333, 0.0043093
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030323, 0.0030155
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032511, 0.0032330
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021356, 0.0021475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016727, upper bound: 0.0017649
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016776, upper bound: 0.0017505
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011698, 0.0011486
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029685, 0.0029148
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018417, 0.0018083
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0033766, 0.0034389
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030195, 0.0029648
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011437, 0.0011230
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0042854, 0.0043644
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029987, 0.0030540
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032151, 0.0032744
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021629, 0.0021237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0012198, upper bound: 0.0012349
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0012198, upper bound: 0.0012349
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011361, 0.0011746
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028829, 0.0029808
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017886, 0.0018493
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034531, 0.0033397
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029324, 0.0030320
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011107, 0.0011484
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043825, 0.0042386
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030666, 0.0029659
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032879, 0.0031799
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021005, 0.0021719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0012273, upper bound: 0.0012144
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0012273, upper bound: 0.0012144
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011492, 0.0011632
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029163, 0.0029517
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018093, 0.0018313
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034195, 0.0033783
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029663, 0.0030024
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011236, 0.0011372
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043397, 0.0042875
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030367, 0.0030002
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032559, 0.0032167
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021248, 0.0021507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017523, upper bound: 0.0017218
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017666, upper bound: 0.0017120
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012536, 0.0012603
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031811, 0.0031981
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019735, 0.0019841
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037049, 0.0036851
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032357, 0.0032530
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012256, 0.0012322
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047020, 0.0046769
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032902, 0.0032727
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035276, 0.0035088
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023178, 0.0023302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0016798
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0016798
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012538, 0.0012601
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031817, 0.0031978
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019739, 0.0019839
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037045, 0.0036859
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032363, 0.0032527
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012258, 0.0012320
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047015, 0.0046778
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032899, 0.0032733
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035273, 0.0035095
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023182, 0.0023300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017481, upper bound: 0.0017775
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017642, upper bound: 0.0017694
time: 1.73 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016179, upper bound: 0.0016523
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0015851, upper bound: 0.0016778
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016583, upper bound: 0.0017270
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016584, upper bound: 0.0017274
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017575, upper bound: 0.0017239
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017575, upper bound: 0.0017239
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0017439
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017406, upper bound: 0.0017439
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0015636, upper bound: 0.0015241
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0015636, upper bound: 0.0015241
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016371, upper bound: 0.0015899
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016368, upper bound: 0.0015899
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016264, upper bound: 0.0015842
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0015741, upper bound: 0.0016385
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016394, upper bound: 0.0016684
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016540, upper bound: 0.0016535
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0015360, upper bound: 0.0015049
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0015360, upper bound: 0.0015049
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0015616, upper bound: 0.0016194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0015753, upper bound: 0.0016034
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0015927, upper bound: 0.0016020
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0015898, upper bound: 0.0016042
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017980, upper bound: 0.0017543
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017683, upper bound: 0.0017830
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017655, upper bound: 0.0017807
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017656, upper bound: 0.0017806
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0012911, upper bound: 0.0012888
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0012911, upper bound: 0.0012888
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017640, upper bound: 0.0017644
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017795, upper bound: 0.0017512
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017824, upper bound: 0.0017676
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017827, upper bound: 0.0017676
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017202, upper bound: 0.0016153
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017013, upper bound: 0.0016220
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016366, upper bound: 0.0016884
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016490, upper bound: 0.0016741
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016917, upper bound: 0.0016055
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017020, upper bound: 0.0015954
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016542, upper bound: 0.0016790
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016445, upper bound: 0.0016907
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017051, upper bound: 0.0017205
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016943, upper bound: 0.0017364
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016921, upper bound: 0.0017355
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017065, upper bound: 0.0017210
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016762, upper bound: 0.0016501
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016895, upper bound: 0.0016371
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016055, upper bound: 0.0017149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016175, upper bound: 0.0016992
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017631, upper bound: 0.0016809
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017374, upper bound: 0.0017122
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017647, upper bound: 0.0017232
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017741, upper bound: 0.0017127
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016727, upper bound: 0.0017649
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016776, upper bound: 0.0017505
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0012198, upper bound: 0.0012349
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0012198, upper bound: 0.0012349
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0012273, upper bound: 0.0012144
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0012273, upper bound: 0.0012144
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017523, upper bound: 0.0017218
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017666, upper bound: 0.0017120
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0016798
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0016798
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017481, upper bound: 0.0017775
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 7, lower bound: -0.0017642, upper bound: 0.0017694

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010339, 0.0010407
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026237, 0.0026410
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016277, 0.0016385
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0030595, 0.0030394
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0026687, 0.0026864
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010108, 0.0010175
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0038829, 0.0038574
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027171, 0.0026992
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029131, 0.0028940
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019116, 0.0019243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0009927, upper bound: 0.0009959
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0009927, upper bound: 0.0009959
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010436, 0.0010300
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026482, 0.0026138
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016430, 0.0016216
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0030280, 0.0030678
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0026937, 0.0026587
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010203, 0.0010070
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0038429, 0.0038934
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0026891, 0.0027244
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0028831, 0.0029210
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019295, 0.0019044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0013789, upper bound: 0.0014695
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0013759, upper bound: 0.0014725
time: 1.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011391, 0.0011113
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028907, 0.0028201
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017934, 0.0017496
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032670, 0.0033488
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029404, 0.0028685
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011137, 0.0010865
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041462, 0.0042500
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029013, 0.0029740
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031106, 0.0031886
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021062, 0.0020548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014587, upper bound: 0.0015148
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014536, upper bound: 0.0015189
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011394, 0.0011112
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028915, 0.0028199
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017939, 0.0017495
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032667, 0.0033496
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029411, 0.0028683
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011140, 0.0010864
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041459, 0.0042511
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029011, 0.0029747
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031104, 0.0031894
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021068, 0.0020546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016125, upper bound: 0.0016492
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015809, upper bound: 0.0016752
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011757, 0.0012044
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029834, 0.0030564
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018509, 0.0018962
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035407, 0.0034561
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030346, 0.0031089
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011494, 0.0011776
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044936, 0.0043863
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031444, 0.0030693
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033713, 0.0032908
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021737, 0.0022269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015525, upper bound: 0.0015116
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015470, upper bound: 0.0015177
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011858, 0.0011918
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030092, 0.0030243
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018669, 0.0018763
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035035, 0.0034861
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030609, 0.0030762
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011594, 0.0011652
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044464, 0.0044243
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031113, 0.0030959
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033359, 0.0033193
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021926, 0.0022035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015525, upper bound: 0.0015116
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015468, upper bound: 0.0015177
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011860, 0.0011926
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030095, 0.0030264
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018671, 0.0018776
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035059, 0.0034864
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030612, 0.0030784
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011595, 0.0011660
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044495, 0.0044247
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031135, 0.0030962
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033382, 0.0033196
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021928, 0.0022051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015310, upper bound: 0.0015341
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015255, upper bound: 0.0015402
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011980, 0.0011814
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030402, 0.0029980
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018861, 0.0018600
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034730, 0.0035219
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030924, 0.0030494
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011713, 0.0011550
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044077, 0.0044698
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030843, 0.0031277
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033068, 0.0033534
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022151, 0.0021844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017330, upper bound: 0.0017369
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017329, upper bound: 0.0017377
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011367, 0.0011868
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028846, 0.0030117
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017896, 0.0018685
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034890, 0.0033417
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029341, 0.0030634
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011114, 0.0011603
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044279, 0.0042410
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030985, 0.0029677
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033220, 0.0031818
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021018, 0.0021944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016168, upper bound: 0.0015509
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015917, upper bound: 0.0015651
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011501, 0.0011747
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029187, 0.0029810
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018107, 0.0018494
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034534, 0.0033811
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029688, 0.0030322
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011245, 0.0011485
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043828, 0.0042911
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030668, 0.0030027
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032881, 0.0032194
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021266, 0.0021720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015365, upper bound: 0.0014970
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015365, upper bound: 0.0014970
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011863, 0.0011595
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030105, 0.0029423
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018677, 0.0018254
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034085, 0.0034875
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030622, 0.0029928
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011599, 0.0011336
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043258, 0.0044261
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030270, 0.0030971
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032454, 0.0033206
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021935, 0.0021438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015919, upper bound: 0.0015626
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016057, upper bound: 0.0015512
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012020, 0.0011467
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030503, 0.0029099
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018924, 0.0018053
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0033710, 0.0035336
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031027, 0.0029599
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011752, 0.0011211
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0042783, 0.0044847
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029937, 0.0031381
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032097, 0.0033646
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022225, 0.0021202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015319, upper bound: 0.0015601
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014928, upper bound: 0.0015896
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012435, 0.0011977
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031556, 0.0030395
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019577, 0.0018857
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035211, 0.0036556
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032098, 0.0030916
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012158, 0.0011710
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044687, 0.0046394
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031270, 0.0032464
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033526, 0.0034807
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022992, 0.0022146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015479, upper bound: 0.0015674
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015479, upper bound: 0.0015674
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012401, 0.0012024
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031470, 0.0030511
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019524, 0.0018929
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035346, 0.0036457
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032010, 0.0031035
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012125, 0.0011755
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044859, 0.0046268
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031390, 0.0032376
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033655, 0.0034712
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022930, 0.0022231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016099, upper bound: 0.0016118
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016129, upper bound: 0.0016108
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011729, 0.0011166
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029764, 0.0028334
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018466, 0.0017579
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032824, 0.0034480
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030275, 0.0028821
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011467, 0.0010916
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041658, 0.0043760
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029150, 0.0030621
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031254, 0.0032831
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021686, 0.0020645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015576, upper bound: 0.0016165
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015576, upper bound: 0.0016167
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011339, 0.0011625
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028773, 0.0029500
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017851, 0.0018302
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034174, 0.0033332
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029267, 0.0030007
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011086, 0.0011366
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043372, 0.0042303
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030349, 0.0029602
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032539, 0.0031738
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020965, 0.0021494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017344, upper bound: 0.0016912
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017344, upper bound: 0.0016902
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011436, 0.0011521
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029021, 0.0029236
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018005, 0.0018138
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0033868, 0.0033620
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029519, 0.0029738
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011181, 0.0011264
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0042983, 0.0042668
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030078, 0.0029857
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032248, 0.0032011
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021145, 0.0021302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017242, upper bound: 0.0016710
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016512, upper bound: 0.0017342
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011911, 0.0012021
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030225, 0.0030505
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018752, 0.0018925
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035339, 0.0035014
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030744, 0.0031029
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011645, 0.0011753
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044849, 0.0044438
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031383, 0.0031095
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033648, 0.0033339
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022022, 0.0022226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017194, upper bound: 0.0017052
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016914, upper bound: 0.0017344
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011922, 0.0012011
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030254, 0.0030480
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018770, 0.0018910
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035310, 0.0035048
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030774, 0.0031003
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011656, 0.0011743
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044813, 0.0044481
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031358, 0.0031125
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033620, 0.0033371
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022044, 0.0022208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017456, upper bound: 0.0017453
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017286, upper bound: 0.0017598
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012143, 0.0012181
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030813, 0.0030911
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019117, 0.0019178
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035809, 0.0035696
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031342, 0.0031442
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011872, 0.0011909
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045447, 0.0045303
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031801, 0.0031701
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034096, 0.0033988
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022451, 0.0022522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017350, upper bound: 0.0017355
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017345, upper bound: 0.0017354
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012151, 0.0012186
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030835, 0.0030923
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019130, 0.0019185
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035823, 0.0035721
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031365, 0.0031454
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011880, 0.0011914
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045465, 0.0045335
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031814, 0.0031723
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034109, 0.0034012
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022467, 0.0022531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017311, upper bound: 0.0016404
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016776, upper bound: 0.0017104
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011663, 0.0011810
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029596, 0.0029970
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018362, 0.0018593
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034719, 0.0034286
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030105, 0.0030485
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011403, 0.0011547
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044063, 0.0043513
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030833, 0.0030449
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033058, 0.0032646
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021564, 0.0021836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017456, upper bound: 0.0017472
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017619, upper bound: 0.0017336
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011785, 0.0011685
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029905, 0.0029651
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018553, 0.0018396
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034350, 0.0034644
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030419, 0.0030160
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011522, 0.0011424
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043594, 0.0043967
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030505, 0.0030766
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032706, 0.0032986
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021789, 0.0021604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017456, upper bound: 0.0017471
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017624, upper bound: 0.0017338
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011757, 0.0012288
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029836, 0.0031182
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018510, 0.0019346
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036123, 0.0034564
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030348, 0.0031718
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011495, 0.0012014
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045845, 0.0043866
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032080, 0.0030695
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034395, 0.0032910
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021739, 0.0022720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014017, upper bound: 0.0013135
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014017, upper bound: 0.0013135
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011883, 0.0012162
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030155, 0.0030863
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018708, 0.0019147
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035753, 0.0034933
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030673, 0.0031393
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011618, 0.0011891
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045375, 0.0044335
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031751, 0.0031023
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034043, 0.0033262
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021971, 0.0022487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016736, upper bound: 0.0015953
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016733, upper bound: 0.0015926
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011606, 0.0011665
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029453, 0.0029602
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018273, 0.0018365
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034292, 0.0034120
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029958, 0.0030110
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011347, 0.0011405
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043521, 0.0043302
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030454, 0.0030301
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032652, 0.0032487
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021460, 0.0021568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0016611
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016096, upper bound: 0.0016612
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011579, 0.0011704
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029383, 0.0029702
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018230, 0.0018427
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034408, 0.0034039
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029888, 0.0030212
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011321, 0.0011443
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043668, 0.0043200
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030557, 0.0030229
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032762, 0.0032411
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021409, 0.0021641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016210, upper bound: 0.0016457
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016222, upper bound: 0.0016464
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011233, 0.0011620
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028504, 0.0029488
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017684, 0.0018295
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034161, 0.0033021
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028994, 0.0029995
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010982, 0.0011361
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043355, 0.0041908
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030337, 0.0029325
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032527, 0.0031441
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020769, 0.0021486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016768, upper bound: 0.0015818
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016530, upper bound: 0.0015883
time: 1.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011246, 0.0011641
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028538, 0.0029541
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017705, 0.0018327
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034221, 0.0033060
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029028, 0.0030048
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010995, 0.0011381
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043431, 0.0041957
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030391, 0.0029360
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032584, 0.0031478
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020793, 0.0021524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017016, upper bound: 0.0015951
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017005, upper bound: 0.0015943
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011924, 0.0012129
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030259, 0.0030780
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018773, 0.0019096
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035657, 0.0035054
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030779, 0.0031308
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011658, 0.0011859
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045253, 0.0044488
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031666, 0.0031131
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033951, 0.0033377
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022047, 0.0022427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014599, upper bound: 0.0014756
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014599, upper bound: 0.0014756
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012045, 0.0012000
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030567, 0.0030451
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018964, 0.0018892
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035276, 0.0035410
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031091, 0.0030974
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011777, 0.0011732
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044770, 0.0044940
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031328, 0.0031447
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033589, 0.0033716
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022271, 0.0022187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016165, upper bound: 0.0016630
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016177, upper bound: 0.0016630
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011338, 0.0011618
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028772, 0.0029481
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017850, 0.0018290
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034153, 0.0033331
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029266, 0.0029988
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011085, 0.0011358
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043344, 0.0042302
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030330, 0.0029601
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032519, 0.0031737
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020964, 0.0021481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016600, upper bound: 0.0016881
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016722, upper bound: 0.0016708
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011460, 0.0011507
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029081, 0.0029201
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018042, 0.0018116
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0033828, 0.0033689
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029581, 0.0029702
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011204, 0.0011250
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0042932, 0.0042756
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030042, 0.0029919
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032209, 0.0032078
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021189, 0.0021276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0013824, upper bound: 0.0014265
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0013824, upper bound: 0.0014265
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011269, 0.0011135
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028597, 0.0028256
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017742, 0.0017530
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032733, 0.0033129
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029088, 0.0028741
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011018, 0.0010886
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041542, 0.0042045
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029069, 0.0029421
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031167, 0.0031544
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020836, 0.0020587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014810, upper bound: 0.0015125
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014810, upper bound: 0.0015125
time: 1.40 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0009927, upper bound: 0.0009959
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0009927, upper bound: 0.0009959
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0013789, upper bound: 0.0014695
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0013759, upper bound: 0.0014725
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0014587, upper bound: 0.0015148
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0014536, upper bound: 0.0015189
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016125, upper bound: 0.0016492
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015809, upper bound: 0.0016752
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015525, upper bound: 0.0015116
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015470, upper bound: 0.0015177
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015525, upper bound: 0.0015116
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015468, upper bound: 0.0015177
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015310, upper bound: 0.0015341
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015255, upper bound: 0.0015402
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017330, upper bound: 0.0017369
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017329, upper bound: 0.0017377
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016168, upper bound: 0.0015509
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015917, upper bound: 0.0015651
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015365, upper bound: 0.0014970
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015365, upper bound: 0.0014970
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015919, upper bound: 0.0015626
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016057, upper bound: 0.0015512
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015319, upper bound: 0.0015601
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0014928, upper bound: 0.0015896
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015479, upper bound: 0.0015674
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015479, upper bound: 0.0015674
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016099, upper bound: 0.0016118
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016129, upper bound: 0.0016108
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015576, upper bound: 0.0016165
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0015576, upper bound: 0.0016167
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017344, upper bound: 0.0016912
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017344, upper bound: 0.0016902
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017242, upper bound: 0.0016710
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016512, upper bound: 0.0017342
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017194, upper bound: 0.0017052
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016914, upper bound: 0.0017344
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017456, upper bound: 0.0017453
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017286, upper bound: 0.0017598
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017350, upper bound: 0.0017355
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017345, upper bound: 0.0017354
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017311, upper bound: 0.0016404
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016776, upper bound: 0.0017104
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017456, upper bound: 0.0017472
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017619, upper bound: 0.0017336
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017456, upper bound: 0.0017471
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017624, upper bound: 0.0017338
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0014017, upper bound: 0.0013135
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0014017, upper bound: 0.0013135
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016736, upper bound: 0.0015953
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016733, upper bound: 0.0015926
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0016611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016096, upper bound: 0.0016612
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016210, upper bound: 0.0016457
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016222, upper bound: 0.0016464
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016768, upper bound: 0.0015818
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016530, upper bound: 0.0015883
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017016, upper bound: 0.0015951
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0017005, upper bound: 0.0015943
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0014599, upper bound: 0.0014756
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0014599, upper bound: 0.0014756
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016165, upper bound: 0.0016630
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016177, upper bound: 0.0016630
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016600, upper bound: 0.0016881
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0016722, upper bound: 0.0016708
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0013824, upper bound: 0.0014265
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0013824, upper bound: 0.0014265
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0014810, upper bound: 0.0015125
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.99
Output dim: 7, lower bound: -0.0014810, upper bound: 0.0015125
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0017065, upper bound: 0.0017210
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0016762, upper bound: 0.0016501
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0016895, upper bound: 0.0016371
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0016055, upper bound: 0.0017149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0016175, upper bound: 0.0016992
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0017631, upper bound: 0.0016809
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0017374, upper bound: 0.0017122
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0017647, upper bound: 0.0017232
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0017741, upper bound: 0.0017127
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0016727, upper bound: 0.0017649
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0016776, upper bound: 0.0017505
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0017523, upper bound: 0.0017218
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0017666, upper bound: 0.0017120
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0016798
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0016798
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0017481, upper bound: 0.0017775
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 7, lower bound: -0.0017642, upper bound: 0.0017694

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.00 + 598.46 = 602.46 seconds
