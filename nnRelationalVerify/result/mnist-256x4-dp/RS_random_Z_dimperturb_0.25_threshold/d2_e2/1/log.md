## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.10584259


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502)
1: (-0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242)
2: (-0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177)
3: (-0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0828764, 0.0828763)
4: (-0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526)
5: (-0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356)
6: (-0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796)
7: (-0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079)
8: (0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2490153, 0.2490156)
9: (0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.37 = 2.66 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1453435, upper bound: 0.1453435

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1383037, upper bound: 0.1429985
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1429985, upper bound: 0.1383037
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.06 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.06
Output dim: 8, lower bound: -0.1383037, upper bound: 0.1429985
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.06
Output dim: 8, lower bound: -0.1429985, upper bound: 0.1383037

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0826373, 0.0825737
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2471237, 0.2475064
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1298041, upper bound: 0.1357603
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1298668, upper bound: 0.1337264
time: 0.55 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0825736, 0.0826373
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2475061, 0.2471240
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1104611, upper bound: 0.0942629
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0940471, upper bound: 0.1104248
time: 0.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 8, lower bound: -0.1298041, upper bound: 0.1357603
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 8, lower bound: -0.1298668, upper bound: 0.1337264
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 8, lower bound: -0.1104611, upper bound: 0.0942629
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 8, lower bound: -0.0940471, upper bound: 0.1104248

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0817769, 0.0816071
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2411754, 0.2421949
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1267754, upper bound: 0.1063479
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1028297, upper bound: 0.1331663
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0816708, 0.0816209
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2412579, 0.2415578
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1295147, upper bound: 0.0725602
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0741770, upper bound: 0.1333841
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0822235, 0.0827841
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2483821, 0.2450149
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1097901, upper bound: 0.0920043
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1011116, upper bound: 0.0920310
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0825736, 0.0822872
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2453971, 0.2471240
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0919359, upper bound: 0.1054581
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0916560, upper bound: 0.1094129
time: 0.49 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.11 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 8, lower bound: -0.1267754, upper bound: 0.1063479
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 8, lower bound: -0.1028297, upper bound: 0.1331663
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 8, lower bound: -0.1295147, upper bound: 0.0725602
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 8, lower bound: -0.0741770, upper bound: 0.1333841
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 8, lower bound: -0.1097901, upper bound: 0.0920043
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 8, lower bound: -0.1011116, upper bound: 0.0920310
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 8, lower bound: -0.0919359, upper bound: 0.1054581
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 8, lower bound: -0.0916560, upper bound: 0.1094129

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0813350, 0.0816741
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2417512, 0.2397139
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1255465, upper bound: 0.1053470
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1169415, upper bound: 0.1036815
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0817623, 0.0812713
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2393312, 0.2422810
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0920712, upper bound: 0.0819012
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0824901, upper bound: 0.1005436
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0816700, 0.0819490
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2433734, 0.2416973
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1266900, upper bound: 0.0679882
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1165076, upper bound: 0.0691403
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0820319, 0.0816064
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2413149, 0.2438715
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0729360, upper bound: 0.1227352
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0729791, upper bound: 0.1321845
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0812714, 0.0817623
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2422810, 0.2393312
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1089060, upper bound: 0.0910237
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1076710, upper bound: 0.0910939
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0819492, 0.0817959
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2424169, 0.2433376
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0823083, upper bound: 0.0996003
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0822981, upper bound: 0.0996003
time: 0.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 8, lower bound: -0.1255465, upper bound: 0.1053470
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 8, lower bound: -0.1169415, upper bound: 0.1036815
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 8, lower bound: -0.0920712, upper bound: 0.0819012
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 8, lower bound: -0.0824901, upper bound: 0.1005436
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 8, lower bound: -0.1266900, upper bound: 0.0679882
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 8, lower bound: -0.1165076, upper bound: 0.0691403
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 8, lower bound: -0.0729360, upper bound: 0.1227352
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 8, lower bound: -0.0729791, upper bound: 0.1321845
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 8, lower bound: -0.1089060, upper bound: 0.0910237
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 8, lower bound: -0.1076710, upper bound: 0.0910939
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 8, lower bound: -0.0823083, upper bound: 0.0996003
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 8, lower bound: -0.0822981, upper bound: 0.0996003

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0827190, 0.0827845
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2483373, 0.2479439
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1228042, upper bound: 0.1001607
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1125349, upper bound: 0.1019458
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0828483, 0.0826447
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2474976, 0.2487197
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1136537, upper bound: 0.0983187
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1094767, upper bound: 0.1002521
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0817959, 0.0819491
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2433376, 0.2424166
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1254284, upper bound: 0.0668040
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1195636, upper bound: 0.0662621
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0820225, 0.0817323
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2420344, 0.2437782
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1152649, upper bound: 0.0679121
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1133072, upper bound: 0.0678922
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0827190, 0.0827845
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2483373, 0.2479439
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0697514, upper bound: 0.1150637
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0669687, upper bound: 0.1192660
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0828483, 0.0826447
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2474976, 0.2487197
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0697928, upper bound: 0.1168366
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0678312, upper bound: 0.1294590
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0826448, 0.0828482
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2487197, 0.2474973
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0997957, upper bound: 0.0815282
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0997957, upper bound: 0.0817136
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0827846, 0.0827191
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2479439, 0.2483373
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 154

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1075862, upper bound: 0.0707755
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0707515, upper bound: 0.0908517
time: 0.49 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.1228042, upper bound: 0.1001607
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.1125349, upper bound: 0.1019458
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.1136537, upper bound: 0.0983187
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.1094767, upper bound: 0.1002521
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.1254284, upper bound: 0.0668040
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.1195636, upper bound: 0.0662621
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.1152649, upper bound: 0.0679121
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.1133072, upper bound: 0.0678922
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.0697514, upper bound: 0.1150637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.0669687, upper bound: 0.1192660
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.0697928, upper bound: 0.1168366
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.0678312, upper bound: 0.1294590
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.0997957, upper bound: 0.0815282
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.0997957, upper bound: 0.0817136
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.1075862, upper bound: 0.0707755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 8, lower bound: -0.0707515, upper bound: 0.0908517

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0817959, 0.0819491
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2433376, 0.2424166
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 116

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1110830, upper bound: 0.0895210
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0960708, upper bound: 0.0903405
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0820225, 0.0817323
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2420344, 0.2437782
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 116

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1122380, upper bound: 0.0642340
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0627229, upper bound: 0.1016070
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0817959, 0.0819491
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2433376, 0.2424166
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0967988, upper bound: 0.0787015
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0795751, upper bound: 0.0896755
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0820225, 0.0817323
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2420344, 0.2437782
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0936685, upper bound: 0.0793304
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0789004, upper bound: 0.0904857
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0827190, 0.0827845
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2483373, 0.2479439
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0987164, upper bound: 0.0607149
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0812276, upper bound: 0.0608646
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0828483, 0.0826447
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2474976, 0.2487197
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1161618, upper bound: 0.0611660
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1002565, upper bound: 0.0620442
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0827190, 0.0827845
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2483373, 0.2479439
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 250

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0946142, upper bound: 0.0616691
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0806433, upper bound: 0.0620863
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0828483, 0.0826447
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2474976, 0.2487197
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0944649, upper bound: 0.0614008
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0811751, upper bound: 0.0620741
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0817959, 0.0819491
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2433376, 0.2424166
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0634542, upper bound: 0.0812888
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0626795, upper bound: 0.0950528
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0820225, 0.0817323
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2420344, 0.2437782
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0577199, upper bound: 0.1062280
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0576478, upper bound: 0.1095129
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0817959, 0.0819491
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2433376, 0.2424166
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 116

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0656689, upper bound: 0.0975972
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0643778, upper bound: 0.1135098
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0820225, 0.0817323
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2420344, 0.2437782
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0632035, upper bound: 0.0980096
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0631599, upper bound: 0.1269009
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0816063, 0.0820320
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2438712, 0.2413149
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1065449, upper bound: 0.0676947
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1028764, upper bound: 0.0684498
time: 0.51 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.1110830, upper bound: 0.0895210
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0960708, upper bound: 0.0903405
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.1122380, upper bound: 0.0642340
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0627229, upper bound: 0.1016070
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0967988, upper bound: 0.0787015
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0795751, upper bound: 0.0896755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0936685, upper bound: 0.0793304
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0789004, upper bound: 0.0904857
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0987164, upper bound: 0.0607149
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0812276, upper bound: 0.0608646
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.1161618, upper bound: 0.0611660
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.1002565, upper bound: 0.0620442
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0946142, upper bound: 0.0616691
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0806433, upper bound: 0.0620863
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0944649, upper bound: 0.0614008
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0811751, upper bound: 0.0620741
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0634542, upper bound: 0.0812888
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0626795, upper bound: 0.0950528
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0577199, upper bound: 0.1062280
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0576478, upper bound: 0.1095129
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0656689, upper bound: 0.0975972
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0643778, upper bound: 0.1135098
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0632035, upper bound: 0.0980096
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.0631599, upper bound: 0.1269009
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.1065449, upper bound: 0.0676947
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 8, lower bound: -0.1028764, upper bound: 0.0684498

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0818599, 0.0823064
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2455187, 0.2428365
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 116

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862688, upper bound: 0.0659568
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0678703, upper bound: 0.0794890
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0816700, 0.0819490
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2433734, 0.2416973
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1012974, upper bound: 0.0554839
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0931853, upper bound: 0.0554869
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0813350, 0.0816741
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2417512, 0.2397139
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1050472, upper bound: 0.0521945
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0949145, upper bound: 0.0521944
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0818599, 0.0823064
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2455187, 0.2428365
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0517013, upper bound: 0.0679514
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0506593, upper bound: 0.0838144
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0823699, 0.0817735
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2423182, 0.2459009
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0514647, upper bound: 0.0701211
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0498632, upper bound: 0.0877338
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0817623, 0.0812713
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2393312, 0.2422810
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 116

### Candidate
type: RSZ, layer: 3, pos: 250

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0556059, upper bound: 0.0990631
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0555780, upper bound: 0.1031602
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0817623, 0.0812713
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2393312, 0.2422810
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 250

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0546348, upper bound: 0.1037516
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0546761, upper bound: 0.1156333
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0817322, 0.0820225
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2437782, 0.2420342
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0961023, upper bound: 0.0586454
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0919550, upper bound: 0.0587105
time: 0.49 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0862688, upper bound: 0.0659568
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0678703, upper bound: 0.0794890
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.1012974, upper bound: 0.0554839
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0931853, upper bound: 0.0554869
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.1050472, upper bound: 0.0521945
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0949145, upper bound: 0.0521944
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0517013, upper bound: 0.0679514
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0506593, upper bound: 0.0838144
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0514647, upper bound: 0.0701211
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0498632, upper bound: 0.0877338
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0556059, upper bound: 0.0990631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0555780, upper bound: 0.1031602
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0546348, upper bound: 0.1037516
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0546761, upper bound: 0.1156333
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0961023, upper bound: 0.0586454
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 8, lower bound: -0.0919550, upper bound: 0.0587105

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0566207, 0.0719295, -0.0566207, 0.0719295, -0.1285502, 0.1285502
1: -0.0222155, 0.0228086, -0.0222155, 0.0228086, -0.0450242, 0.0450242
2: -0.0121037, 0.0416140, -0.0121037, 0.0416140, -0.0537177, 0.0537177
3: -0.0156537, 0.0756290, -0.0156537, 0.0756290, -0.0823699, 0.0817735
4: -0.0289626, 0.0032900, -0.0289626, 0.0032900, -0.0322526, 0.0322526
5: -0.0056761, 0.0525595, -0.0056761, 0.0525595, -0.0582356, 0.0582356
6: -0.0459933, 0.0623863, -0.0459933, 0.0623863, -0.1083796, 0.1083796
7: -0.0243630, 0.0181449, -0.0243630, 0.0181449, -0.0425079, 0.0425079
8: 0.6653527, 0.9646113, 0.6653527, 0.9646113, -0.2423182, 0.2459009
9: 0.0385295, 0.0965836, 0.0385295, 0.0965836, -0.0580541, 0.0580541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 250

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0511271, upper bound: 0.0675974
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0500939, upper bound: 0.0874031
time: 0.55 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 8, lower bound: -0.0511271, upper bound: 0.0675974
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 8, lower bound: -0.0500939, upper bound: 0.0874031

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.66 + 93.51 = 96.17 seconds
