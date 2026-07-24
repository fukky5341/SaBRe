## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.294357096
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.8942013, 1.8942013)
1: (2.4124451, 3.9131384, 2.4124451, 3.9131384, -1.4570764, 1.4570764)
2: (-6.5181541, -4.9777999, -6.5181541, -4.9777999, -1.4703059, 1.4703059)
3: (-11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.8738117, 1.8738117)
4: (-4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.5592403, 1.5592403)
5: (-12.3431225, -10.5792007, -12.3431225, -10.5792007, -1.6700945, 1.6700947)
6: (-10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.9091916, 1.9091921)
7: (-4.2142544, -2.6923499, -4.2142544, -2.6923499, -1.3958604, 1.3958603)
8: (-3.2913580, -1.8388863, -3.2913580, -1.8388863, -1.3433269, 1.3433267)
9: (-12.0051117, -10.4397650, -12.0051117, -10.4397650, -1.5272553, 1.5272552)

## BASE Result
execution time: IAR + LP analysis = 15.15 + 32.07 = 47.23 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.77 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.1101056337356567

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=0.9366202354431152
rel_dist={1: [-0.5181458851188188, 0.518142788570696]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=0.8209632635116577
rel_dist={1: [-0.29885889706970215, 0.2988555407171436]}

## Binary Search Result
Binary search time: 148.82 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 3403.95 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=None

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5176339, upper bound: 0.5181415
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181450, upper bound: 0.5176328
time: 4.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.16 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.16
Output dim: 1, lower bound: -0.5176339, upper bound: 0.5181415
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.16
Output dim: 1, lower bound: -0.5181450, upper bound: 0.5176328

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2394595, 1.2373700
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9332358, 0.9345483
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9353073, 0.9316094
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.1127567, 1.1175925
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.1033356, 1.1048675
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9396586, 0.9403347
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1121182, 1.1164864
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8212776, 0.8239460
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.7143631, 0.7094470
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8809700, 0.8758345

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5113348, upper bound: 0.5181347
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5113348, upper bound: 0.5118751
time: 5.25 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2373700, 1.2394595
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9345483, 0.9332358
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9316092, 0.9353074
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.1175923, 1.1127568
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.1048677, 1.1033356
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9403348, 0.9396584
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1164865, 1.1121184
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8239460, 0.8212776
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.7094469, 0.7143632
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8758345, 0.8809700

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5143066, upper bound: 0.5169589
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5174655, upper bound: 0.5138001
time: 5.01 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.13
Output dim: 1, lower bound: -0.5113348, upper bound: 0.5181347
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.13
Output dim: 1, lower bound: -0.5113348, upper bound: 0.5118751
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.13
Output dim: 1, lower bound: -0.5143066, upper bound: 0.5169589
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.13
Output dim: 1, lower bound: -0.5174655, upper bound: 0.5138001

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2390552, 1.2362437
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9266485, 0.9321530
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9324112, 0.9236782
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.1053743, 1.1149080
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0964096, 1.1023474
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9352789, 0.9387360
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1107697, 1.1159948
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8206365, 0.8237128
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.7120702, 0.7031451
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8787475, 0.8697177

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5041299, upper bound: 0.5181326
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5113268, upper bound: 0.5109282
time: 3.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2383332, 1.2369661
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9308425, 0.9279611
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9273763, 0.9287174
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.1100709, 1.1102099
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.1008179, 1.0979416
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9380612, 0.9359552
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1116281, 1.1151378
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8210442, 0.8233051
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.7080613, 0.7071555
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8748531, 0.8736142

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5137934, upper bound: 0.5111956
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5169523, upper bound: 0.5080365
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2357612, 1.2360692
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9342718, 0.9326531
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9302912, 0.9325390
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.1170490, 1.1116138
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0991025, 1.1005971
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9399772, 0.9389157
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1158400, 1.1107563
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8229878, 0.8208209
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.7087663, 0.7129446
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8747489, 0.8787258

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5105504, upper bound: 0.5169532
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5105897, upper bound: 0.5100404
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2339797, 1.2378507
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9339657, 0.9329591
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9288409, 0.9339893
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.1164494, 1.1122134
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.1021295, 1.0975704
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9395921, 0.9393009
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1151242, 1.1114719
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8234894, 0.8203193
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.7080286, 0.7136825
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8735900, 0.8798845

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111959, upper bound: 0.5137960
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5174586, upper bound: 0.5075020
time: 4.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.90 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.5041299, upper bound: 0.5181326
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.5113268, upper bound: 0.5109282
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.5137934, upper bound: 0.5111956
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.5169523, upper bound: 0.5080365
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.5105504, upper bound: 0.5169532
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.5105897, upper bound: 0.5100404
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.5111959, upper bound: 0.5137960
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.5174586, upper bound: 0.5075020

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2240272, 1.2154436
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8988749, 0.9120787
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9206069, 0.9073422
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0719974, 1.0687079
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0483789, 1.0676473
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9331727, 0.9372144
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0959344, 1.1052755
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8052342, 0.8023973
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6980921, 0.6838075
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8516397, 0.8322338

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5037460, upper bound: 0.5174523
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5037453, upper bound: 0.5105818
time: 3.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2182550, 1.2212155
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9065741, 0.9043794
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9160752, 0.9118737
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0591743, 1.0815310
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0617099, 1.0543165
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9337575, 0.9366297
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1000504, 1.1011597
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7993212, 0.8083104
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6927325, 0.6891670
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8412633, 0.8426100

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5037853, upper bound: 0.5105393
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5106558, upper bound: 0.5105429
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2367251, 1.2335763
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9305658, 0.9273784
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9260581, 0.9259491
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.1095276, 1.1090668
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0950527, 1.0952032
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9377036, 0.9352125
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1109819, 1.1137757
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8200860, 0.8228486
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.7073807, 0.7057368
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8737679, 0.8713698

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5100373, upper bound: 0.5111901
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5100766, upper bound: 0.5042772
time: 5.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2349436, 1.2353580
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9302597, 0.9276843
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9246078, 0.9273994
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.1089280, 1.1096666
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0980797, 1.0921764
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9373186, 0.9355975
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1102662, 1.1144915
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8205879, 0.8223469
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.7066427, 0.7064747
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8726091, 0.8725287

Time for backsubstitution: 15.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5100340, upper bound: 0.5043196
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5169470, upper bound: 0.5042808
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2207394, 1.2152762
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9064980, 0.9125787
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9184887, 0.9162052
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0836723, 1.0654138
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0510819, 1.0659077
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9378715, 0.9373947
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1010046, 1.1000369
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8075855, 0.7995054
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6947932, 0.6936071
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8476627, 0.8412484

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5042807, upper bound: 0.5169496
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5105435, upper bound: 0.5106556
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2149682, 1.2210484
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9135866, 0.9048793
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9139576, 0.9207368
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0708492, 1.0782368
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0644124, 1.0525765
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9376822, 0.9368100
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1036882, 1.0959210
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8016722, 0.8044139
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6894288, 0.6989666
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8372717, 0.8516246

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5043200, upper bound: 0.5100365
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5105828, upper bound: 0.5037425
time: 4.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2335765, 1.2367249
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9273787, 0.9305658
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9259491, 0.9260581
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.1090667, 1.1095276
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0952034, 1.0950525
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9352126, 0.9377036
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1137757, 1.1109817
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8228486, 0.8200861
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.7057369, 0.7073807
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8713698, 0.8737677

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5042799, upper bound: 0.5100770
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111906, upper bound: 0.5100398
time: 4.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2328541, 1.2374473
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9315705, 0.9263718
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9209099, 0.9310932
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.1137650, 1.1048310
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0996089, 1.0906446
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9379935, 0.9349214
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1146326, 1.1101234
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8232565, 0.8196784
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.7017266, 0.7113894
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8674731, 0.8776619

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5105402, upper bound: 0.5037850
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5174533, upper bound: 0.5037461
time: 4.06 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5037460, upper bound: 0.5174523
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5037453, upper bound: 0.5105818
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5037853, upper bound: 0.5105393
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5106558, upper bound: 0.5105429
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5100373, upper bound: 0.5111901
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5100766, upper bound: 0.5042772
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5100340, upper bound: 0.5043196
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5169470, upper bound: 0.5042808
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5042807, upper bound: 0.5169496
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5105435, upper bound: 0.5106556
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5043200, upper bound: 0.5100365
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5105828, upper bound: 0.5037425
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5042799, upper bound: 0.5100770
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5111906, upper bound: 0.5100398
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5105402, upper bound: 0.5037850
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 1, lower bound: -0.5174533, upper bound: 0.5037461

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2224240, 1.2120602
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8985982, 0.9114960
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9192908, 0.9045759
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0714543, 1.0675648
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0426240, 1.0649195
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9328153, 0.9364718
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0952883, 1.1039135
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8042760, 0.8019408
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6974163, 0.6823889
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8505754, 0.8299960

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1964

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2615

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4925392, upper bound: 0.5051708
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4920679, upper bound: 0.5056560
time: 6.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2206435, 1.2138419
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8982921, 0.9111912
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9178407, 0.9060262
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0708544, 1.0681645
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0456505, 1.0618923
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9324300, 0.9360828
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0945725, 1.1031971
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8037732, 0.8014390
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6966735, 0.6831268
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8494020, 0.8311548

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1403

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2536

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4885015, upper bound: 0.4919451
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4858906, upper bound: 0.4990677
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2166533, 1.2178321
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9056869, 0.9037967
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9147594, 0.9091074
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0586309, 1.0803879
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0559545, 1.0515883
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9326260, 0.9358871
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0979719, 1.0997977
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7983630, 0.8068492
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6920519, 0.6877486
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8401847, 0.8403722

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1403

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4796047, upper bound: 0.5063021
time: 6.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5001738, upper bound: 0.4883211
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2148719, 1.2196126
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9059916, 0.9041026
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9133091, 0.9105575
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0580313, 1.0809877
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0589819, 1.0485615
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9330149, 0.9362721
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0986886, 1.1005135
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7988646, 0.8073523
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6913140, 0.6884912
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8390257, 0.8415458

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 723

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5073018, upper bound: 0.5072849
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5072863, upper bound: 0.5072877
time: 4.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2217021, 1.2127829
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9027920, 0.9073040
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9142556, 0.9096153
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0761507, 1.0628668
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0470319, 1.0605139
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9355974, 0.9336909
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0961466, 1.1030566
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8046837, 0.8015330
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6934074, 0.6863992
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8466816, 0.8338925

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 740

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5079890, upper bound: 0.5111878
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5100364, upper bound: 0.5091401
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2159314, 1.2185547
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9098806, 0.8996047
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9097245, 0.9141468
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0633276, 1.0756899
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0603623, 1.0471827
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9354081, 0.9331062
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0988302, 1.0989407
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7987707, 0.8064415
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6880430, 0.6917588
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8362904, 0.8442687

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1164

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5053149, upper bound: 0.4997114
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5056833, upper bound: 0.4993020
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2199216, 1.2145646
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9024863, 0.9069993
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9128056, 0.9110656
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0755510, 1.0634665
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0500588, 1.0574867
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9352124, 0.9333020
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0954309, 1.1023402
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8041809, 0.8010314
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6926647, 0.6871371
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8455079, 0.8350513

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 723

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 740

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5079863, upper bound: 0.5043209
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5100331, upper bound: 0.5022698
time: 4.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2141500, 1.2203352
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9101853, 0.8999107
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9082739, 0.9155968
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0627282, 1.0762897
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0633898, 1.0441558
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9357970, 0.9334912
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0995469, 1.0996565
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7992723, 0.8069445
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6873051, 0.6925015
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8351316, 0.8454423

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2340

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 227

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5148316, upper bound: 0.5021187
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5147933, upper bound: 0.5021284
time: 4.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2203355, 1.2141497
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8999109, 0.9101856
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9155970, 0.9082739
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0762899, 1.0627279
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0441556, 1.0633900
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9334915, 0.9357970
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0996566, 1.0995468
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8069446, 0.7992723
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6925015, 0.6873051
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8454423, 0.8351316

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2879

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1745

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5006542, upper bound: 0.5158343
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5032298, upper bound: 0.5115888
time: 4.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2196126, 1.2148719
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9041028, 0.9059916
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9105575, 0.9133091
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0809877, 1.0580312
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0485616, 1.0589819
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9362724, 0.9330148
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1005135, 1.0986885
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8073523, 0.7988646
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6884912, 0.6913140
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8415458, 0.8390257

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 2615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2536

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4990292, upper bound: 0.4928011
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4919033, upper bound: 0.4954148
time: 5.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2145648, 1.2199216
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9069996, 0.9024861
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9110656, 0.9128056
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0634665, 1.0755510
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0574865, 1.0500587
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9333022, 0.9352123
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1023402, 1.0954310
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8010314, 0.8041807
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6871371, 0.6926647
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8350513, 0.8455079

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 675

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 697

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5032745, upper bound: 0.5090404
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5033327, upper bound: 0.5089855
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2138419, 1.2206438
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9111915, 0.8982922
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9060264, 0.9178406
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0681646, 1.0708543
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0618920, 1.0456507
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9360831, 0.9324300
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1031971, 1.0945727
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8014390, 0.8037730
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6831268, 0.6966735
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8311548, 0.8494020

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 570

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 227

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5084356, upper bound: 0.5015660
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5084132, upper bound: 0.5015894
time: 4.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2185550, 1.2159314
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8996048, 0.9098808
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9141467, 0.9097242
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0756900, 1.0633276
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0471826, 1.0603627
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9331062, 0.9354080
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0989408, 1.0988305
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8064415, 0.7987705
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6917588, 0.6880430
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8442688, 0.8362904

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1984

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1781

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5018900, upper bound: 0.5076779
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5018875, upper bound: 0.5076778
time: 3.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2127829, 1.2217021
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9073043, 0.9027921
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9096153, 0.9142556
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0628669, 1.0761507
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0605140, 1.0470320
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9336910, 0.9355973
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1030564, 1.0961467
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8015330, 0.8046837
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6863992, 0.6934074
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8338923, 0.8466815

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5100644, upper bound: 0.5095477
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5107081, upper bound: 0.5089172
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2178321, 1.2166536
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9037967, 0.9056869
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9091074, 0.9147594
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0803881, 1.0586309
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0515881, 1.0559547
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9358871, 0.9326259
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0997977, 1.0979720
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8068492, 0.7983629
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6877486, 0.6920518
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8403721, 0.8401846

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1978

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2489

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5083925, upper bound: 0.5037839
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5105371, upper bound: 0.5016430
time: 3.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2120605, 1.2224243
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9114962, 0.8985982
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9045761, 0.9192907
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0675650, 1.0714540
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0649195, 1.0426239
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9364719, 0.9328151
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1039133, 1.0952883
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8019407, 0.8042760
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6823889, 0.6974163
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8299961, 0.8505756

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 570

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5141951, upper bound: 0.5003770
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5141951, upper bound: 0.5003911
time: 4.26 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.4925392, upper bound: 0.5051708
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.4920679, upper bound: 0.5056560
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.4885015, upper bound: 0.4919451
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.4858906, upper bound: 0.4990677
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.4796047, upper bound: 0.5063021
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5001738, upper bound: 0.4883211
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5073018, upper bound: 0.5072849
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5072863, upper bound: 0.5072877
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5079890, upper bound: 0.5111878
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5100364, upper bound: 0.5091401
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5053149, upper bound: 0.4997114
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5056833, upper bound: 0.4993020
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5079863, upper bound: 0.5043209
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5100331, upper bound: 0.5022698
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5148316, upper bound: 0.5021187
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5147933, upper bound: 0.5021284
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5006542, upper bound: 0.5158343
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5032298, upper bound: 0.5115888
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.4990292, upper bound: 0.4928011
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.4919033, upper bound: 0.4954148
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5032745, upper bound: 0.5090404
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5033327, upper bound: 0.5089855
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5084356, upper bound: 0.5015660
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5084132, upper bound: 0.5015894
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5018900, upper bound: 0.5076779
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5018875, upper bound: 0.5076778
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5100644, upper bound: 0.5095477
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5107081, upper bound: 0.5089172
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5083925, upper bound: 0.5037839
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5105371, upper bound: 0.5016430
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5141951, upper bound: 0.5003770
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 1, lower bound: -0.5141951, upper bound: 0.5003911

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2198753, 1.2079420
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8988318, 0.9113570
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9202020, 0.9027090
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0705106, 1.0656266
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0427423, 1.0642838
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9256377, 0.9507228
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0930309, 1.1066984
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8056866, 0.8008894
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6920449, 0.6798238
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8494480, 0.8296323

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1997

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 675

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4908001, upper bound: 0.5016501
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4890494, upper bound: 0.5034952
time: 3.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2224240, 1.2095110
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8984592, 0.9114960
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9174240, 0.9045759
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0714543, 1.0666214
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0419884, 1.0649195
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9328153, 0.9292946
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0952883, 1.1016561
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8032247, 0.8019408
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6948512, 0.6823889
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8502119, 0.8299960

Time for backsubstitution: 14.46 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=0.9366202354431152
rel_dist={1: [-0.5181458851188188, 0.518142788570696]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2975559, upper bound: 0.2986063
time: 7.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2975559, upper bound: 0.2975539
time: 4.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.28
Output dim: 1, lower bound: -0.2975559, upper bound: 0.2986063
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.28
Output dim: 1, lower bound: -0.2975559, upper bound: 0.2975539

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0727167, 1.0721231
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8204827, 0.8203806
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8212733, 0.8207898
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9579415, 0.9577415
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9965885, 0.9975972
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7788544, 0.7787257
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9476523, 0.9474137
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7010467, 0.7012138
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5829334, 0.5826875
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7454441, 0.7450579

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2963010, upper bound: 0.2986060
time: 7.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2963139, upper bound: 0.2962993
time: 6.39 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0721231, 1.0727167
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8203807, 0.8204826
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8207898, 0.8212733
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9577415, 0.9579414
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9975970, 0.9965882
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7787259, 0.7788541
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9474137, 0.9476523
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7012138, 0.7010466
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5826875, 0.5829334
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7450578, 0.7454442

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2963021, upper bound: 0.2963162
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2986081, upper bound: 0.2962997
time: 6.11 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.81 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.81
Output dim: 1, lower bound: -0.2963010, upper bound: 0.2986060
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.81
Output dim: 1, lower bound: -0.2963139, upper bound: 0.2962993
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.81
Output dim: 1, lower bound: -0.2963021, upper bound: 0.2963162
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.81
Output dim: 1, lower bound: -0.2986081, upper bound: 0.2962997

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0538473, 1.0513296
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7927089, 0.7951735
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8064501, 0.8044562
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9160159, 0.9115417
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9485674, 0.9540198
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7767482, 0.7768148
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9328172, 0.9339505
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6817021, 0.6798983
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5653839, 0.5633498
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7114305, 0.7075806

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941799, upper bound: 0.2986035
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2962989, upper bound: 0.2964842
time: 6.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0519238, 1.0532537
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7950721, 0.7926071
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8049397, 0.8059669
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9117415, 0.9158161
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9530110, 0.9495761
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7766852, 0.7766199
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9337118, 0.9325786
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6797311, 0.6815344
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5635958, 0.5651363
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7079668, 0.7110393

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2962913, upper bound: 0.2962991
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2963139, upper bound: 0.2962766
time: 4.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0532537, 1.0519235
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7926073, 0.7950720
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8059669, 0.8049397
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9158158, 0.9117415
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9495759, 0.9530108
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7766199, 0.7766851
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9325786, 0.9337118
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6815345, 0.6797310
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5651363, 0.5635958
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7110393, 0.7079669

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941810, upper bound: 0.2963107
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2941810, upper bound: 0.2941951
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0513296, 1.0538471
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7951736, 0.7927090
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8044562, 0.8064501
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9115415, 0.9160159
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9540200, 0.9485672
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7768149, 0.7767482
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9339507, 0.9328172
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6798983, 0.6817021
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5633498, 0.5653839
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7075806, 0.7114305

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964870, upper bound: 0.2963000
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2986038, upper bound: 0.2941795
time: 5.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.09
Output dim: 1, lower bound: -0.2941799, upper bound: 0.2986035
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.09
Output dim: 1, lower bound: -0.2962989, upper bound: 0.2964842
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.09
Output dim: 1, lower bound: -0.2962913, upper bound: 0.2962991
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.09
Output dim: 1, lower bound: -0.2963139, upper bound: 0.2962766
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.09
Output dim: 1, lower bound: -0.2941810, upper bound: 0.2963107
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 24.09
Output dim: 1, lower bound: -0.2941810, upper bound: 0.2941951
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.09
Output dim: 1, lower bound: -0.2964870, upper bound: 0.2963000
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.09
Output dim: 1, lower bound: -0.2986038, upper bound: 0.2941795

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0529609, 1.0502026
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7861228, 0.7899853
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8002014, 0.7965276
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9086318, 0.9057230
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9416411, 0.9485631
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7723684, 0.7733624
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9314690, 0.9328885
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6810606, 0.6793926
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5604179, 0.5570470
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7066116, 0.7014627

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941594, upper bound: 0.2986043
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941798, upper bound: 0.2985806
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0527201, 1.0504436
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7875209, 0.7885873
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7985218, 0.7982075
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9101973, 0.9041575
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9431107, 0.9470940
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7732959, 0.7724351
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9317551, 0.9326024
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6811965, 0.6792567
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5590812, 0.5583838
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7053127, 0.7027615

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2962764, upper bound: 0.2964840
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2962989, upper bound: 0.2964613
time: 4.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0472193, 1.0478530
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7916875, 0.7896599
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7966328, 0.7964272
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8992634, 0.9049497
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9490597, 0.9461355
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7749434, 0.7751033
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9224329, 0.9227560
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6728480, 0.6755410
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5525354, 0.5524372
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6964014, 0.6977620

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941701, upper bound: 0.2962972
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2962894, upper bound: 0.2941780
time: 4.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0465226, 1.0485494
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7921247, 0.7892224
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7954001, 0.7976599
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9008751, 0.9033377
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9495699, 0.9456248
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7751687, 0.7748780
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9238889, 0.9212998
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6737378, 0.6746515
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5508968, 0.5540760
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6946895, 0.6994739

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941928, upper bound: 0.2962744
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2963118, upper bound: 0.2941552
time: 4.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0523672, 1.0507965
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7860208, 0.7898837
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7997181, 0.7970111
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9084318, 0.9059229
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9426501, 0.9475541
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7722402, 0.7732327
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9312303, 0.9326497
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6808927, 0.6792252
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5601703, 0.5572930
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7062201, 0.7018490

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941582, upper bound: 0.2963125
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941582, upper bound: 0.2962893
time: 7.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0504436, 1.0527201
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7885876, 0.7875208
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7982075, 0.7985218
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9041574, 0.9101973
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9470937, 0.9431105
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7724349, 0.7732959
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9326024, 0.9317552
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6792567, 0.6811963
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5583838, 0.5590811
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7027617, 0.7053127

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964619, upper bound: 0.2962984
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964642, upper bound: 0.2962786
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0502024, 1.0529609
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7899852, 0.7861229
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7965279, 0.8002014
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9057231, 0.9086317
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9485633, 0.9416413
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7733624, 0.7723684
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9328885, 0.9314691
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6793926, 0.6810604
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5570470, 0.5604179
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7014627, 0.7066115

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2985812, upper bound: 0.2941792
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2986038, upper bound: 0.2941593
time: 4.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2941594, upper bound: 0.2986043
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2941798, upper bound: 0.2985806
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2962764, upper bound: 0.2964840
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2962989, upper bound: 0.2964613
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2941701, upper bound: 0.2962972
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2962894, upper bound: 0.2941780
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2941928, upper bound: 0.2962744
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2963118, upper bound: 0.2941552
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2941582, upper bound: 0.2963125
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2941582, upper bound: 0.2962893
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2964619, upper bound: 0.2962984
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2964642, upper bound: 0.2962786
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2985812, upper bound: 0.2941792
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.2986038, upper bound: 0.2941593

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0482574, 1.0448022
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7827373, 0.7870365
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7918904, 0.7869854
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8961551, 0.8948587
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9376898, 0.9451218
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7706263, 0.7718453
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9201901, 0.9230651
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6741784, 0.6733999
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5493579, 0.5443487
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6950462, 0.6881864

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1164

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2913103, upper bound: 0.2961008
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2919266, upper bound: 0.2957728
time: 7.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0475607, 1.0454988
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7831750, 0.7865998
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7906592, 0.7882180
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8977671, 0.8932464
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9382005, 0.9446120
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7708519, 0.7716203
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9216461, 0.9216095
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6750679, 0.6725104
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5477196, 0.5459875
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6933351, 0.6898983

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 740

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2936587, upper bound: 0.2969177
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2924775, upper bound: 0.2980629
time: 7.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0480161, 1.0450432
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7841353, 0.7856392
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7902119, 0.7886651
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8977208, 0.8932927
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9391589, 0.9436533
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7715538, 0.7709184
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9204762, 0.9227794
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6743143, 0.6732640
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5480217, 0.5456855
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6937482, 0.6894853

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 697

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2958872, upper bound: 0.2961038
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2959054, upper bound: 0.2961085
time: 8.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0473199, 1.0457397
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7845721, 0.7852017
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7889793, 0.7898965
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8993332, 0.8916808
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9396691, 0.9431427
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7717788, 0.7706929
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9219317, 0.9213234
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6752038, 0.6723745
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5463829, 0.5473238
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6920364, 0.6911963

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1794

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2902429, upper bound: 0.2936236
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2934601, upper bound: 0.2904050
time: 4.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0463333, 1.0467262
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7851000, 0.7844701
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7903800, 0.7884960
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8918808, 0.8991331
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9421334, 0.9406781
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7705634, 0.7716503
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9210846, 0.9216932
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6722074, 0.6750362
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5475698, 0.5461353
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6915827, 0.6916451

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1984

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2615

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2925831, upper bound: 0.2926179
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2904914, upper bound: 0.2947061
time: 4.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0460925, 1.0469670
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7864981, 0.7830728
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7887015, 0.7901757
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8934464, 0.8975670
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9436030, 0.9392095
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7714908, 0.7707233
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9213707, 0.9214075
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6723433, 0.6749003
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5462335, 0.5474721
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6902845, 0.6929440

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 723

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2933421, upper bound: 0.2912724
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2933421, upper bound: 0.2912723
time: 5.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0456371, 1.0474229
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7855377, 0.7840332
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7891486, 0.7897286
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8934927, 0.8975208
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9426441, 0.9401684
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7707887, 0.7714254
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9225407, 0.9202375
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6730969, 0.6741467
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5459315, 0.5477740
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6898715, 0.6933570

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 767

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2939340, upper bound: 0.2960752
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2939934, upper bound: 0.2960156
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0453963, 1.0476637
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7869349, 0.7826353
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7874689, 0.7914069
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8950589, 0.8959552
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9441128, 0.9386988
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7717156, 0.7704980
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9228263, 0.9199514
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6732328, 0.6740108
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5445948, 0.5491103
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6885726, 0.6946551

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1403

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1978

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2949173, upper bound: 0.2927371
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2948933, upper bound: 0.2927606
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0476637, 1.0453963
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7826352, 0.7869350
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7914071, 0.7874689
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8959553, 0.8950586
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9386988, 0.9441128
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7704980, 0.7717156
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9199514, 0.9228263
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6740108, 0.6732328
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5491103, 0.5445947
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6946549, 0.6885726

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 723

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2615

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2925689, upper bound: 0.2926320
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2904797, upper bound: 0.2947220
time: 4.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0469670, 1.0460927
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7830729, 0.7864981
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7901757, 0.7887015
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8975670, 0.8934463
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9392095, 0.9436030
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7707236, 0.7714906
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9214075, 0.9213707
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6749003, 0.6723433
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5474721, 0.5462335
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6929440, 0.6902845

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 570

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 912

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2929652, upper bound: 0.2942586
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2921360, upper bound: 0.2951520
time: 4.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0457397, 1.0473199
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7852015, 0.7845720
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7898965, 0.7889793
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8916810, 0.8993330
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9431424, 0.9396691
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7706931, 0.7717787
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9213233, 0.9219317
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6723745, 0.6752038
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5473238, 0.5463829
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6911964, 0.6920364

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1845

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964588, upper bound: 0.2956256
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2957893, upper bound: 0.2962951
time: 4.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0450435, 1.0480163
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7856393, 0.7841352
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7886653, 0.7902119
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8932927, 0.8977207
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9436531, 0.9391594
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7709184, 0.7715538
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9227796, 0.9204762
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6732640, 0.6743143
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5456855, 0.5480216
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6894853, 0.6937482

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 697

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1164

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2936618, upper bound: 0.2940174
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2940155, upper bound: 0.2934522
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0454988, 1.0475607
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7865996, 0.7831748
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7882183, 0.7906592
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8932464, 0.8977669
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9446120, 0.9382005
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7716205, 0.7708517
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9216094, 0.9216461
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6725104, 0.6750679
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5459875, 0.5477196
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6898983, 0.6933352

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 227

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2984873, upper bound: 0.2911153
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2955175, upper bound: 0.2940851
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0448022, 1.0482571
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7870364, 0.7827373
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7869854, 0.7918904
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8948588, 0.8961551
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9451218, 0.9376900
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7718453, 0.7706263
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9230652, 0.9201901
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6733999, 0.6741784
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5443488, 0.5493579
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6881864, 0.6950463

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2982944, upper bound: 0.2884201
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2928672, upper bound: 0.2938475
time: 4.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2913103, upper bound: 0.2961008
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2919266, upper bound: 0.2957728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2936587, upper bound: 0.2969177
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2924775, upper bound: 0.2980629
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2958872, upper bound: 0.2961038
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2959054, upper bound: 0.2961085
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2902429, upper bound: 0.2936236
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2934601, upper bound: 0.2904050
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2925831, upper bound: 0.2926179
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2904914, upper bound: 0.2947061
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2933421, upper bound: 0.2912724
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2933421, upper bound: 0.2912723
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2939340, upper bound: 0.2960752
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2939934, upper bound: 0.2960156
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2949173, upper bound: 0.2927371
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2948933, upper bound: 0.2927606
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2925689, upper bound: 0.2926320
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2904797, upper bound: 0.2947220
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2929652, upper bound: 0.2942586
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2921360, upper bound: 0.2951520
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2964588, upper bound: 0.2956256
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2957893, upper bound: 0.2962951
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2936618, upper bound: 0.2940174
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2940155, upper bound: 0.2934522
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2984873, upper bound: 0.2911153
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2955175, upper bound: 0.2940851
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2982944, upper bound: 0.2884201
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.56
Output dim: 1, lower bound: -0.2928672, upper bound: 0.2938475

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0465949, 1.0437417
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7784581, 0.7824333
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7895877, 0.7857354
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8960464, 0.8947723
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9368348, 0.9443021
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7663771, 0.7684143
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9193087, 0.9223503
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6739371, 0.6730410
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5483025, 0.5436075
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6928158, 0.6879327

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1397

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2908056, upper bound: 0.2952738
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2904833, upper bound: 0.2955967
time: 4.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0471961, 1.0433249
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7786946, 0.7827573
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7906401, 0.7851348
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8960688, 0.8947619
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9368701, 0.9444356
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7671953, 0.7679667
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9196067, 0.9221836
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6740081, 0.6731585
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5486168, 0.5434552
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6957865, 0.6859560

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 662

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 912

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2902030, upper bound: 0.2936632
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2892671, upper bound: 0.2942402
time: 4.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0501387, 1.0479932
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7829171, 0.7859844
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7909822, 0.7885199
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8720267, 0.8669925
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9427299, 0.9487042
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7701569, 0.7710645
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9219193, 0.9213762
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6313527, 0.6279414
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5484457, 0.5467266
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7002523, 0.6977714

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1164
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1164

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2896049, upper bound: 0.2921013
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2911849, upper bound: 0.2940874
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0500500, 1.0480766
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7825490, 0.7863420
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7909603, 0.7885411
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8715129, 0.8674767
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9422927, 0.9491343
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7702894, 0.7709253
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9214029, 0.9218827
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6304989, 0.6287490
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5484588, 0.5467112
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7011817, 0.6968155

Time for backsubstitution: 14.44 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=0.8209632635116577
rel_dist={1: [-0.29885889706970215, 0.2988555407171436]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1665.66 seconds
