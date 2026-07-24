## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.22581355849999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7738495, 0.7738495)
1: (-11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6459391, 0.6459394)
2: (-7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6083186, 0.6083186)
3: (-5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6005569, 0.6005573)
4: (-7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8191080, 0.8191080)
5: (5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5779729, 0.5779729)
6: (-9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8672638, 0.8672638)
7: (-14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7276466, 0.7276464)
8: (-3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6108687, 0.6108685)
9: (-6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6705360, 0.6705360)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.21 + 33.82 = 58.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2269478, upper bound: 0.2269479

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4576

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4610

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269434, upper bound: 0.2269268
time: 5.14 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269271, upper bound: 0.2269435
time: 3.81 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.96 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.96
Output dim: 5, lower bound: -0.2269434, upper bound: 0.2269268
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.96
Output dim: 5, lower bound: -0.2269271, upper bound: 0.2269435

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7764368, 0.7736878
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6498094, 0.6456983
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6119039, 0.6080949
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6005411, 0.6008110
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8203015, 0.8190346
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5789185, 0.5779152
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8734951, 0.8668747
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7321920, 0.7273660
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6110258, 0.6108582
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6702242, 0.6755395

Time for backsubstitution: 22.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4576
type: DSZ, layer: 1, pos: 4571

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269413, upper bound: 0.2269270
time: 5.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269428, upper bound: 0.2269254
time: 4.43 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7736878, 0.7738495
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6456981, 0.6459394
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6080949, 0.6083186
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6005569, 0.6005416
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8190346, 0.8191080
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5779152, 0.5779729
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8668752, 0.8672638
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7273660, 0.7276464
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6108584, 0.6108685
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6705360, 0.6702242

Time for backsubstitution: 22.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4576
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4571

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4576

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269162, upper bound: 0.2266805
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266637, upper bound: 0.2269330
time: 3.59 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.24 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.24
Output dim: 5, lower bound: -0.2269413, upper bound: 0.2269270
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.24
Output dim: 5, lower bound: -0.2269428, upper bound: 0.2269254
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.24
Output dim: 5, lower bound: -0.2269162, upper bound: 0.2266805
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.24
Output dim: 5, lower bound: -0.2266637, upper bound: 0.2269330

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7764373, 0.7736950
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6496859, 0.6479039
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6118417, 0.6091692
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6018848, 0.6007376
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8202095, 0.8206275
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5791373, 0.5779054
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8734317, 0.8679557
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7321839, 0.7275300
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6109900, 0.6114790
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6745970, 0.6752868

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4576

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4571

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269274, upper bound: 0.2266710
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266852, upper bound: 0.2269131
time: 3.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7764368, 0.7736878
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6498094, 0.6455748
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6119039, 0.6080327
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6004686, 0.6008110
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8203015, 0.8189428
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5789084, 0.5779152
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8734951, 0.8668118
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7321920, 0.7273579
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6110258, 0.6108222
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6699712, 0.6755395

Time for backsubstitution: 22.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4576

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4571

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269290, upper bound: 0.2266694
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266868, upper bound: 0.2269112
time: 3.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7738361, 0.7738185
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6456952, 0.6459532
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6079490, 0.6090040
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6003284, 0.6016083
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8203902, 0.8188210
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5791826, 0.5777020
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8666925, 0.8681107
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7284122, 0.7274241
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6108217, 0.6110382
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6704564, 0.6705973

Time for backsubstitution: 23.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4571

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269141, upper bound: 0.2266800
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269157, upper bound: 0.2266784
time: 4.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7736568, 0.7738495
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6456981, 0.6459363
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6080949, 0.6081727
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6005569, 0.6003122
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8187466, 0.8191080
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5776434, 0.5779729
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8668752, 0.8670812
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7271442, 0.7276464
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6108584, 0.6108317
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6705360, 0.6701434

Time for backsubstitution: 23.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4571

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266616, upper bound: 0.2269325
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266632, upper bound: 0.2269308
time: 3.65 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.46 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.46
Output dim: 5, lower bound: -0.2269274, upper bound: 0.2266710
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.46
Output dim: 5, lower bound: -0.2266852, upper bound: 0.2269131
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.46
Output dim: 5, lower bound: -0.2269290, upper bound: 0.2266694
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.46
Output dim: 5, lower bound: -0.2266868, upper bound: 0.2269112
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.46
Output dim: 5, lower bound: -0.2269141, upper bound: 0.2266800
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.46
Output dim: 5, lower bound: -0.2269157, upper bound: 0.2266784
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.46
Output dim: 5, lower bound: -0.2266616, upper bound: 0.2269325
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.46
Output dim: 5, lower bound: -0.2266632, upper bound: 0.2269308

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7763119, 0.7749519
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6521406, 0.6476612
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6128161, 0.6090727
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6060719, 0.6003256
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8213153, 0.8205178
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5817051, 0.5776513
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8733702, 0.8685799
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7337065, 0.7273793
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6136200, 0.6112185
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6777301, 0.6749752

Time for backsubstitution: 24.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4576

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4576

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269165, upper bound: 0.2264076
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266640, upper bound: 0.2266601
time: 4.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7764373, 0.7735696
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6494427, 0.6479039
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6117451, 0.6091692
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6014724, 0.6007376
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8200998, 0.8206275
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5788841, 0.5779054
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8734317, 0.8678942
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7320337, 0.7275300
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6107290, 0.6114790
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6742854, 0.6752868

Time for backsubstitution: 23.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4576

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4576

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266743, upper bound: 0.2266497
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264218, upper bound: 0.2269022
time: 3.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7763119, 0.7749453
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6522646, 0.6453321
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6128783, 0.6079361
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6046548, 0.6003981
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8214068, 0.8188331
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5814762, 0.5776603
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8734326, 0.8674364
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7337151, 0.7272072
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6136563, 0.6105616
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6731033, 0.6752276

Time for backsubstitution: 23.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4576

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4576

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269181, upper bound: 0.2264060
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266656, upper bound: 0.2266585
time: 3.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7764368, 0.7735624
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6495667, 0.6455748
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6118073, 0.6080327
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6000562, 0.6008110
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8201914, 0.8189428
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5786543, 0.5779152
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8734951, 0.8667498
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7320414, 0.7273579
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6107652, 0.6108222
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6696601, 0.6755395

Time for backsubstitution: 23.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4576

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4576

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266759, upper bound: 0.2266481
time: 4.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264234, upper bound: 0.2269002
time: 3.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7738361, 0.7738256
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6455717, 0.6481586
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6078861, 0.6100779
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6016717, 0.6015358
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8202987, 0.8204141
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5794024, 0.5776930
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8666301, 0.8691921
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7284040, 0.7275887
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6107860, 0.6116595
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6748288, 0.6703446

Time for backsubstitution: 23.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4571

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4571

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269000, upper bound: 0.2264236
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266581, upper bound: 0.2266661
time: 3.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7738361, 0.7738190
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6456952, 0.6458294
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6079490, 0.6089413
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6002555, 0.6016083
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8203902, 0.8187294
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5791731, 0.5777020
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8666925, 0.8680482
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7284122, 0.7274160
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6108217, 0.6110027
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6702034, 0.6705973

Time for backsubstitution: 23.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4571

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4571

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269016, upper bound: 0.2264224
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266596, upper bound: 0.2266645
time: 3.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7736568, 0.7738566
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6455750, 0.6481416
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6080327, 0.6092465
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6019001, 0.6002398
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8186550, 0.8207009
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5778632, 0.5779641
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8668118, 0.8681626
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7271361, 0.7278109
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6108222, 0.6114531
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6749089, 0.6698906

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4571

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4571

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266475, upper bound: 0.2266765
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264056, upper bound: 0.2269182
time: 3.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7736568, 0.7738500
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6456981, 0.6458125
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6080949, 0.6081100
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6004839, 0.6003122
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8187466, 0.8190162
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5776339, 0.5779729
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8668752, 0.8670187
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7271442, 0.7276382
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6108584, 0.6107962
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6702831, 0.6701434

Time for backsubstitution: 23.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4571

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4571

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266491, upper bound: 0.2266749
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264072, upper bound: 0.2269170
time: 3.75 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2269165, upper bound: 0.2264076
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2266640, upper bound: 0.2266601
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2266743, upper bound: 0.2266497
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2264218, upper bound: 0.2269022
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2269181, upper bound: 0.2264060
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2266656, upper bound: 0.2266585
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2266759, upper bound: 0.2266481
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2264234, upper bound: 0.2269002
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2269000, upper bound: 0.2264236
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2266581, upper bound: 0.2266661
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2269016, upper bound: 0.2264224
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2266596, upper bound: 0.2266645
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2266475, upper bound: 0.2266765
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2264056, upper bound: 0.2269182
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2266491, upper bound: 0.2266749
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.24
Output dim: 5, lower bound: -0.2264072, upper bound: 0.2269170

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7764597, 0.7749209
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6521378, 0.6476750
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6126699, 0.6097577
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6058431, 0.6013923
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8226709, 0.8202298
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5829735, 0.5773802
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8731875, 0.8694272
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7347527, 0.7271571
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6135838, 0.6113889
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6776495, 0.6753485

Time for backsubstitution: 23.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 960
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 313
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 617
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 608
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 1795
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 1801
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 771
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 1935

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 423

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264403, upper bound: 0.2242368
time: 5.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2247462, upper bound: 0.2259317
time: 4.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7762809, 0.7749519
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6521406, 0.6476581
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6128161, 0.6089263
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6060719, 0.6000962
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8210273, 0.8205178
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5814342, 0.5776513
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8733702, 0.8683977
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7334847, 0.7273793
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6136200, 0.6111822
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6777301, 0.6748946

Time for backsubstitution: 23.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 617
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 960
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 608
type: DSZ, layer: 3, pos: 1795
type: DSZ, layer: 3, pos: 771
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 313
type: DSZ, layer: 3, pos: 1801

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1492

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2259257, upper bound: 0.2261253
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2261292, upper bound: 0.2259220
time: 3.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7765856, 0.7735386
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6494398, 0.6479177
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6115985, 0.6098542
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6012430, 0.6018047
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8214555, 0.8203397
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5801516, 0.5776339
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8732500, 0.8687410
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7330794, 0.7273083
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6106932, 0.6116495
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6742048, 0.6756599

Time for backsubstitution: 23.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 771
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 313
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 608
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1795
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 960
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 617
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1801

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 709

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2265091, upper bound: 0.2260991
time: 5.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2261101, upper bound: 0.2264840
time: 3.91 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 32.62 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.62
Output dim: 5, lower bound: -0.2264403, upper bound: 0.2242368
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.62
Output dim: 5, lower bound: -0.2247462, upper bound: 0.2259317
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.62
Output dim: 5, lower bound: -0.2259257, upper bound: 0.2261253
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.62
Output dim: 5, lower bound: -0.2261292, upper bound: 0.2259220
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.62
Output dim: 5, lower bound: -0.2265091, upper bound: 0.2260991
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.62
Output dim: 5, lower bound: -0.2261101, upper bound: 0.2264840
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2264218, upper bound: 0.2269022
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2269181, upper bound: 0.2264060
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2266656, upper bound: 0.2266585
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2266759, upper bound: 0.2266481
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2264234, upper bound: 0.2269002
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2269000, upper bound: 0.2264236
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2266581, upper bound: 0.2266661
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2269016, upper bound: 0.2264224
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2266596, upper bound: 0.2266645
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2266475, upper bound: 0.2266765
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2264056, upper bound: 0.2269182
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2266491, upper bound: 0.2266749
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.62
Output dim: 5, lower bound: -0.2264072, upper bound: 0.2269170

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.03 + 542.68 = 600.71 seconds
