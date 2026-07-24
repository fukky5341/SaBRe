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
execution time: IAR + RelationalAnalysis = 23.27 + 34.55 = 57.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2269478, upper bound: 0.2269479

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4576
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4610

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269434, upper bound: 0.2269268
time: 5.19 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269271, upper bound: 0.2269435
time: 3.93 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.36 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.36
Output dim: 5, lower bound: -0.2269434, upper bound: 0.2269268
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.36
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

Time for backsubstitution: 20.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4576
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 4571

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269306, upper bound: 0.2266715
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266874, upper bound: 0.2269146
time: 4.06 seconds

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

Time for backsubstitution: 21.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4576
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 4571

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269141, upper bound: 0.2266876
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266711, upper bound: 0.2269310
time: 4.35 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.04 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.04
Output dim: 5, lower bound: -0.2269306, upper bound: 0.2266715
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.04
Output dim: 5, lower bound: -0.2266874, upper bound: 0.2269146
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.04
Output dim: 5, lower bound: -0.2269141, upper bound: 0.2266876
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.04
Output dim: 5, lower bound: -0.2266711, upper bound: 0.2269310

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7763119, 0.7749457
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6522646, 0.6454556
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6128783, 0.6079984
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6047277, 0.6003981
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8214068, 0.8189247
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5814853, 0.5776603
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8734326, 0.8674989
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7337151, 0.7272153
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6136563, 0.6105978
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6733561, 0.6752276

Time for backsubstitution: 21.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4576
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 4576

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269197, upper bound: 0.2264081
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266672, upper bound: 0.2266600
time: 3.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7764368, 0.7735629
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6495667, 0.6456983
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6118073, 0.6080949
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6001291, 0.6008110
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8201914, 0.8190346
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5786633, 0.5779152
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8734951, 0.8668122
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7320414, 0.7273660
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6107652, 0.6108582
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6699123, 0.6755395

Time for backsubstitution: 21.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4576
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 4576

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266764, upper bound: 0.2266513
time: 4.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264240, upper bound: 0.2269037
time: 3.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7735629, 0.7751074
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6481533, 0.6456964
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6090693, 0.6082218
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6047435, 0.6001291
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8201399, 0.8189991
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5804820, 0.5777197
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8668127, 0.8678875
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7288885, 0.7274957
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6134884, 0.6106079
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6736679, 0.6699126

Time for backsubstitution: 21.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4576
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4576

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269032, upper bound: 0.2264242
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266507, upper bound: 0.2266767
time: 4.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7736878, 0.7737246
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6454554, 0.6459394
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6079984, 0.6083186
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6001444, 0.6005416
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8189249, 0.8191080
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5776601, 0.5779729
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8668752, 0.8672009
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7272158, 0.7276464
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6105978, 0.6108685
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6702242, 0.6702242

Time for backsubstitution: 21.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4576
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4576

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266602, upper bound: 0.2266676
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264077, upper bound: 0.2269199
time: 3.84 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.10 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 5, lower bound: -0.2269197, upper bound: 0.2264081
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 5, lower bound: -0.2266672, upper bound: 0.2266600
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 5, lower bound: -0.2266764, upper bound: 0.2266513
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 5, lower bound: -0.2264240, upper bound: 0.2269037
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 5, lower bound: -0.2269032, upper bound: 0.2264242
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 5, lower bound: -0.2266507, upper bound: 0.2266767
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 5, lower bound: -0.2266602, upper bound: 0.2266676
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.10
Output dim: 5, lower bound: -0.2264077, upper bound: 0.2269199

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7764606, 0.7749147
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6522613, 0.6454694
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6127324, 0.6086838
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6044993, 0.6014652
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8227630, 0.8186369
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5827532, 0.5773888
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8732510, 0.8683467
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7347608, 0.7269931
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6136191, 0.6107676
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6732750, 0.6756010

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269165, upper bound: 0.2264076
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269181, upper bound: 0.2264060
time: 4.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7762814, 0.7749457
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6522646, 0.6454525
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6128783, 0.6078525
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6047277, 0.6001692
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8211193, 0.8189247
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5812135, 0.5776603
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8734326, 0.8673177
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7334929, 0.7272153
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6136563, 0.6105611
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6733561, 0.6751471

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266640, upper bound: 0.2266601
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266656, upper bound: 0.2266585
time: 3.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7765851, 0.7735319
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6495638, 0.6457121
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6116614, 0.6087804
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.5999002, 0.6018777
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8215475, 0.8187468
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5799313, 0.5776434
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8733125, 0.8676605
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7330875, 0.7271438
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6107285, 0.6110282
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6698318, 0.6759126

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266743, upper bound: 0.2266497
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266759, upper bound: 0.2266481
time: 4.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7764063, 0.7735629
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6495667, 0.6456952
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6118073, 0.6079490
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6001291, 0.6005821
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8199039, 0.8190346
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5783920, 0.5779152
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8734951, 0.8666310
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7318192, 0.7273660
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6107652, 0.6108217
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6699123, 0.6754587

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264218, upper bound: 0.2269022
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264234, upper bound: 0.2269002
time: 3.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7737112, 0.7750764
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6481504, 0.6457102
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6089234, 0.6089072
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6045151, 0.6011963
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8214960, 0.8187113
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5817499, 0.5774479
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8666310, 0.8687353
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7299347, 0.7272744
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6134522, 0.6107776
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6735878, 0.6702857

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269000, upper bound: 0.2264236
time: 4.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269016, upper bound: 0.2264224
time: 3.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7735319, 0.7751074
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6481533, 0.6456933
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6090693, 0.6080756
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6047435, 0.5999002
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8198524, 0.8189991
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5802107, 0.5777197
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8668127, 0.8677058
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7286668, 0.7274957
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6134884, 0.6105711
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6736679, 0.6698318

Time for backsubstitution: 21.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266475, upper bound: 0.2266765
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266491, upper bound: 0.2266749
time: 3.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7738361, 0.7736936
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6454525, 0.6459532
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6078525, 0.6090040
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.5999160, 0.6016083
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8202806, 0.8188210
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5789285, 0.5777020
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8666925, 0.8680491
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7282615, 0.7274241
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6105611, 0.6110382
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6701446, 0.6705973

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266581, upper bound: 0.2266661
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266596, upper bound: 0.2266645
time: 3.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7736568, 0.7737246
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6454554, 0.6459363
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6079984, 0.6081727
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6001444, 0.6003122
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8186369, 0.8191080
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5773888, 0.5779729
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8668752, 0.8670197
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7269931, 0.7276464
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6105978, 0.6108317
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6702242, 0.6701434

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264056, upper bound: 0.2269182
time: 4.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264072, upper bound: 0.2269170
time: 3.92 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2269165, upper bound: 0.2264076
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2269181, upper bound: 0.2264060
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2266640, upper bound: 0.2266601
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2266656, upper bound: 0.2266585
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2266743, upper bound: 0.2266497
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2266759, upper bound: 0.2266481
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2264218, upper bound: 0.2269022
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2264234, upper bound: 0.2269002
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2269000, upper bound: 0.2264236
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2269016, upper bound: 0.2264224
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2266475, upper bound: 0.2266765
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2266491, upper bound: 0.2266749
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2266581, upper bound: 0.2266661
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2266596, upper bound: 0.2266645
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 5, lower bound: -0.2264056, upper bound: 0.2269182
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
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

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 771
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 1801
type: DSZ, layer: 3, pos: 313
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1795
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 960
type: DSZ, layer: 3, pos: 617
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 608
type: DSZ, layer: 3, pos: 415

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 711

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2175558, upper bound: 0.2243496
time: 5.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2248593, upper bound: 0.2170469
time: 3.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7764606, 0.7749143
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6522613, 0.6453459
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6127324, 0.6086211
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6044259, 0.6014652
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8227630, 0.8185451
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5827441, 0.5773888
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8732510, 0.8682833
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7347608, 0.7269850
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6136191, 0.6107321
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6730227, 0.6756010

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 771
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 1801
type: DSZ, layer: 3, pos: 313
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1795
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 960
type: DSZ, layer: 3, pos: 617
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 608
type: DSZ, layer: 3, pos: 415

Time for candidate selection: 0.45 seconds

### Candidate
type: DSZ, layer: 3, pos: 711

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2175574, upper bound: 0.2243487
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2248608, upper bound: 0.2170449
time: 4.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 771
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 2522
type: DSZ, layer: 3, pos: 1801
type: DSZ, layer: 3, pos: 313
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1795
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 960
type: DSZ, layer: 3, pos: 617
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 722
type: DSZ, layer: 3, pos: 608
type: DSZ, layer: 3, pos: 415

Time for candidate selection: 0.43 seconds

### Candidate
type: DSZ, layer: 3, pos: 711

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2173034, upper bound: 0.2246028
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2246067, upper bound: 0.2172992
time: 3.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7762814, 0.7749453
1: -11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6522646, 0.6453290
2: -7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6128783, 0.6077898
3: -5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6046548, 0.6001692
4: -7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8211193, 0.8188331
5: 5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5812044, 0.5776603
6: -9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8734326, 0.8672543
7: -14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7334929, 0.7272072
8: -3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6136563, 0.6105254
9: -6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6731033, 0.6751471

Time for backsubstitution: 22.02 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.83 + 543.01 = 600.84 seconds
