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
execution time: IAR + LP analysis = 15.09 + 31.97 = 47.06 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.94 seconds, max iter: 100)

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
Binary search time: 148.08 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 3404.87 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=None

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5176339, upper bound: 0.5181415
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181450, upper bound: 0.5176328
time: 4.17 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.13
Output dim: 1, lower bound: -0.5176339, upper bound: 0.5181415
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.13
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
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5104287, upper bound: 0.5181370
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5176259, upper bound: 0.5109386
time: 4.27 seconds

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

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109386, upper bound: 0.5176283
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109386, upper bound: 0.5104286
time: 4.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.99
Output dim: 1, lower bound: -0.5104287, upper bound: 0.5181370
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.99
Output dim: 1, lower bound: -0.5176259, upper bound: 0.5109386
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.99
Output dim: 1, lower bound: -0.5109386, upper bound: 0.5176283
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.99
Output dim: 1, lower bound: -0.5109386, upper bound: 0.5104286

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2244310, 1.2165694
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9054621, 0.9144740
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9235029, 0.9152734
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0793800, 1.0713925
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0553048, 1.0701675
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9375527, 0.9388136
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0972829, 1.1057669
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8058751, 0.8026304
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.7003851, 0.6901093
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8538625, 0.8383508

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5041274, upper bound: 0.5181326
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5104197, upper bound: 0.5118721
time: 4.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2186589, 1.2223415
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9131616, 0.9067745
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9189715, 0.9198050
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0665569, 1.0842156
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0686357, 1.0568366
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9381375, 0.9382288
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1013989, 1.1016511
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7999620, 0.8085437
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6950254, 0.6954688
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8434863, 0.8487270

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5113268, upper bound: 0.5109282
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5176195, upper bound: 0.5046673
time: 3.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2223415, 1.2186589
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9067746, 0.9131614
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9198050, 0.9189715
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0842156, 1.0665569
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0568368, 1.0686355
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9382291, 0.9381374
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1016512, 1.1013988
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8085437, 0.7999620
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6954689, 0.6950256
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8487267, 0.8434863

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046680, upper bound: 0.5176192
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109292, upper bound: 0.5113265
time: 4.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2165694, 1.2244310
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9144739, 0.9054620
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9152734, 0.9235030
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0713925, 1.0793800
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0701678, 1.0553048
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9388137, 0.9375527
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1057668, 1.0972830
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8026307, 0.8058751
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6901093, 0.7003851
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8383508, 0.8538625

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118696, upper bound: 0.5104192
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181302, upper bound: 0.5041299
time: 4.17 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.85 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.85
Output dim: 1, lower bound: -0.5041274, upper bound: 0.5181326
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.85
Output dim: 1, lower bound: -0.5104197, upper bound: 0.5118721
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.85
Output dim: 1, lower bound: -0.5113268, upper bound: 0.5109282
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.85
Output dim: 1, lower bound: -0.5176195, upper bound: 0.5046673
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.85
Output dim: 1, lower bound: -0.5046680, upper bound: 0.5176192
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.85
Output dim: 1, lower bound: -0.5109292, upper bound: 0.5113265
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.85
Output dim: 1, lower bound: -0.5118696, upper bound: 0.5104192
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.85
Output dim: 1, lower bound: -0.5181302, upper bound: 0.5041299

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

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5037460, upper bound: 0.5174523
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5037453, upper bound: 0.5105818
time: 3.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2233047, 1.2161663
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9030689, 0.9078867
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9155717, 0.9123814
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0766943, 1.0640099
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0527873, 1.0632417
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9359548, 0.9344335
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0967927, 1.1044185
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8056419, 0.8019896
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6940833, 0.6878177
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8477454, 0.8361303

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5100373, upper bound: 0.5111901
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5100340, upper bound: 0.5043196
time: 4.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5037853, upper bound: 0.5105393
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5106558, upper bound: 0.5105429
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2175331, 1.2219381
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9107682, 0.9001874
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9110403, 0.9169130
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0638711, 1.0768330
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0661178, 1.0499110
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9365396, 0.9338487
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1009088, 1.1003027
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7997289, 0.8079026
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6887237, 0.6931773
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8373694, 0.8465065

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5100766, upper bound: 0.5042772
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5169470, upper bound: 0.5042808
time: 3.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2219381, 1.2175331
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9001874, 0.9107683
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9169130, 0.9110402
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0768330, 1.0638710
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0499110, 1.0661178
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9338489, 0.9365395
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1003027, 1.1009088
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8079028, 0.7997286
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6931773, 0.6887237
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8465066, 0.8373693

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5042807, upper bound: 0.5169496
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5042799, upper bound: 0.5100770
time: 3.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2212152, 1.2182553
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9043792, 0.9065742
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9118738, 0.9160752
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0815313, 1.0591743
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0543165, 1.0617099
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9366298, 0.9337573
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1011596, 1.1000504
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8083105, 0.7993209
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6891670, 0.6927325
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8426099, 0.8412634

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5105435, upper bound: 0.5106556
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5105402, upper bound: 0.5037850
time: 4.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2161665, 1.2233050
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9078869, 0.9030688
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9123816, 0.9155717
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0640099, 1.0766940
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0632415, 1.0527871
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9344337, 0.9359548
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1044183, 1.0967929
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8019896, 0.8056419
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6878178, 0.6940832
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8361301, 0.8477455

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5043200, upper bound: 0.5100365
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111906, upper bound: 0.5100398
time: 4.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2154436, 1.2240272
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.9120787, 0.8988749
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9073422, 0.9206069
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0687082, 1.0719974
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0676475, 1.0483789
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9372146, 0.9331726
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1052756, 1.0959346
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.8023973, 0.8052342
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6838075, 0.6980921
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8322339, 0.8516396

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5105828, upper bound: 0.5037425
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5174533, upper bound: 0.5037461
time: 4.02 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5037460, upper bound: 0.5174523
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5037453, upper bound: 0.5105818
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5100373, upper bound: 0.5111901
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5100340, upper bound: 0.5043196
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5037853, upper bound: 0.5105393
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5106558, upper bound: 0.5105429
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5100766, upper bound: 0.5042772
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5169470, upper bound: 0.5042808
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5042807, upper bound: 0.5169496
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5042799, upper bound: 0.5100770
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5105435, upper bound: 0.5106556
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5105402, upper bound: 0.5037850
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5043200, upper bound: 0.5100365
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5111906, upper bound: 0.5100398
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 1, lower bound: -0.5105828, upper bound: 0.5037425
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
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

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4937581, upper bound: 0.5110380
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4973119, upper bound: 0.5074726
time: 4.07 seconds

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

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4937549, upper bound: 0.5041391
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4973085, upper bound: 0.5005831
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5000385, upper bound: 0.5047544
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5036162, upper bound: 0.5011861
time: 4.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5000352, upper bound: 0.4978510
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5036127, upper bound: 0.4943046
time: 4.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4937975, upper bound: 0.5040960
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4973516, upper bound: 0.5005398
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5006835, upper bound: 0.5040993
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5042505, upper bound: 0.5005434
time: 4.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5000777, upper bound: 0.4978077
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5036558, upper bound: 0.4942611
time: 3.97 seconds

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
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5069637, upper bound: 0.4978112
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5105547, upper bound: 0.4942648
time: 4.11 seconds

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

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4942623, upper bound: 0.5105575
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4978127, upper bound: 0.5069663
time: 4.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4942592, upper bound: 0.5036548
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4978095, upper bound: 0.5000803
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5005444, upper bound: 0.5042485
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5040964, upper bound: 0.5006822
time: 3.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5005413, upper bound: 0.4973499
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5040931, upper bound: 0.4937959
time: 5.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4943018, upper bound: 0.5036156
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4978527, upper bound: 0.5000348
time: 4.09 seconds

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

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5011878, upper bound: 0.5036148
time: 7.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5047515, upper bound: 0.5000377
time: 4.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5005838, upper bound: 0.4973068
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5041362, upper bound: 0.4937535
time: 6.46 seconds

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
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5074698, upper bound: 0.4973100
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5110351, upper bound: 0.4937571
time: 6.56 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.4937581, upper bound: 0.5110380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.4973119, upper bound: 0.5074726
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.4937549, upper bound: 0.5041391
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.4973085, upper bound: 0.5005831
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5000385, upper bound: 0.5047544
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5036162, upper bound: 0.5011861
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5000352, upper bound: 0.4978510
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5036127, upper bound: 0.4943046
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.4937975, upper bound: 0.5040960
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.4973516, upper bound: 0.5005398
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5006835, upper bound: 0.5040993
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5042505, upper bound: 0.5005434
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5000777, upper bound: 0.4978077
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5036558, upper bound: 0.4942611
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5069637, upper bound: 0.4978112
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5105547, upper bound: 0.4942648
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.4942623, upper bound: 0.5105575
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.4978127, upper bound: 0.5069663
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.4942592, upper bound: 0.5036548
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.4978095, upper bound: 0.5000803
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5005444, upper bound: 0.5042485
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5040964, upper bound: 0.5006822
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5005413, upper bound: 0.4973499
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5040931, upper bound: 0.4937959
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.4943018, upper bound: 0.5036156
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.4978527, upper bound: 0.5000348
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5011878, upper bound: 0.5036148
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5047515, upper bound: 0.5000377
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5005838, upper bound: 0.4973068
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5041362, upper bound: 0.4937535
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5074698, upper bound: 0.4973100
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.92
Output dim: 1, lower bound: -0.5110351, upper bound: 0.4937571

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2140985, 1.1894472
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8958344, 0.9107834
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9128783, 0.9055597
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0827184, 1.0537658
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0327682, 1.0652947
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9367318, 0.9243416
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.1002021, 1.0987210
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7978338, 0.7970192
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6923592, 0.6752781
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8546009, 0.8277723

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2615

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4060490, upper bound: 0.4167492
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4060490, upper bound: 0.4167492
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.2224240, 1.2037346
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8985982, 0.9087324
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.9192908, 0.8981636
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0576549, 1.0675648
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0426240, 1.0550636
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.9206851, 0.9364718
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0900958, 1.1039135
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7993544, 0.8019408
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6974163, 0.6773318
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8483520, 0.8299960

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2615

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4060490, upper bound: 0.4167487
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4060490, upper bound: 0.4167487
time: 4.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 23.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.36
Output dim: 1, lower bound: -0.4060490, upper bound: 0.4167492
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.36
Output dim: 1, lower bound: -0.4060490, upper bound: 0.4167492
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.36
Output dim: 1, lower bound: -0.4060490, upper bound: 0.4167487
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.36
Output dim: 1, lower bound: -0.4060490, upper bound: 0.4167487
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.4937549, upper bound: 0.5041391
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.4973085, upper bound: 0.5005831
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5000385, upper bound: 0.5047544
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5036162, upper bound: 0.5011861
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5000352, upper bound: 0.4978510
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5036127, upper bound: 0.4943046
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.4937975, upper bound: 0.5040960
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.4973516, upper bound: 0.5005398
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5006835, upper bound: 0.5040993
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5042505, upper bound: 0.5005434
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5000777, upper bound: 0.4978077
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5036558, upper bound: 0.4942611
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5069637, upper bound: 0.4978112
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5105547, upper bound: 0.4942648
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.4942623, upper bound: 0.5105575
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.4978127, upper bound: 0.5069663
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.4942592, upper bound: 0.5036548
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.4978095, upper bound: 0.5000803
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5005444, upper bound: 0.5042485
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5040964, upper bound: 0.5006822
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5005413, upper bound: 0.4973499
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5040931, upper bound: 0.4937959
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.4943018, upper bound: 0.5036156
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.4978527, upper bound: 0.5000348
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5011878, upper bound: 0.5036148
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5047515, upper bound: 0.5000377
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5005838, upper bound: 0.4973068
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5041362, upper bound: 0.4937535
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5074698, upper bound: 0.4973100
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.36
Output dim: 1, lower bound: -0.5110351, upper bound: 0.4937571
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=0.9366202354431152
rel_dist={1: [-0.5181458851188188, 0.518142788570696]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988366, upper bound: 0.2988571
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988366, upper bound: 0.2988344
time: 4.06 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.14
Output dim: 1, lower bound: -0.2988366, upper bound: 0.2988571
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.14
Output dim: 1, lower bound: -0.2988366, upper bound: 0.2988344

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0708084, 1.0701122
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8175789, 0.8180163
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8152514, 0.8140187
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9464064, 0.9480183
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9984019, 0.9989123
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7777267, 0.7779518
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9374971, 0.9389532
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6951222, 0.6960115
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5730455, 0.5714068
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7357366, 0.7340248

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964358, upper bound: 0.2988572
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964358, upper bound: 0.2964583
time: 4.90 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0701122, 1.0708089
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8180164, 0.8175788
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8140187, 0.8152516
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9480183, 0.9464064
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9989121, 0.9984016
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7779520, 0.7777264
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9389532, 0.9374971
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6960115, 0.6951220
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5714068, 0.5730456
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7340248, 0.7357367

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964358, upper bound: 0.2988319
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964358, upper bound: 0.2964329
time: 5.23 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.42
Output dim: 1, lower bound: -0.2964358, upper bound: 0.2988572
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.42
Output dim: 1, lower bound: -0.2964358, upper bound: 0.2964583
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.42
Output dim: 1, lower bound: -0.2964358, upper bound: 0.2988319
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.42
Output dim: 1, lower bound: -0.2964358, upper bound: 0.2964329

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0519323, 1.0493116
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7898053, 0.7928090
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8004262, 0.7976830
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9044809, 0.9018184
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9503710, 0.9553251
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7756207, 0.7760410
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9226618, 0.9254898
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6757777, 0.6746960
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5554944, 0.5520691
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7017117, 0.6965411

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2988552
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2967342
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0500083, 1.0512357
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7923716, 0.7902427
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7989156, 0.7991934
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9002066, 0.9060928
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9548142, 0.9508815
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7758155, 0.7758460
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9240336, 0.9241178
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6738064, 0.6766670
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5537078, 0.5538557
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6982529, 0.6999998

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2964562
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2943372
time: 4.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0512357, 1.0500083
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7902426, 0.7923715
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7991934, 0.7989156
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9060926, 0.9002066
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9508812, 0.9548144
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7758460, 0.7758156
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9241178, 0.9240338
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6766670, 0.6738064
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5538557, 0.5537078
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6999998, 0.6982530

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2988303
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2967107
time: 5.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0493116, 1.0519321
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7928091, 0.7898052
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7976830, 0.8004262
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9018183, 0.9044809
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9553254, 0.9503708
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7760410, 0.7756207
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9254899, 0.9226618
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6746960, 0.6757774
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5520692, 0.5554944
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6965411, 0.7017118

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2964321
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2943133
time: 5.11 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.93
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2988552
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.93
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2967342
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.93
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2964562
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 25.93
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2943372
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.93
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2988303
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.93
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2967107
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.93
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2964321
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 25.93
Output dim: 1, lower bound: -0.2943145, upper bound: 0.2943133

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0510468, 1.0481858
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7832180, 0.7876192
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7941732, 0.7897515
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8970983, 0.8960018
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9434452, 0.9498677
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7712407, 0.7725878
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9213133, 0.9244270
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6751366, 0.6741909
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5505289, 0.5457673
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6968927, 0.6904241

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941571, upper bound: 0.2986031
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941582, upper bound: 0.2963125
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0508060, 1.0484266
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7846159, 0.7862220
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7924950, 0.7914314
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8986640, 0.8944358
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9449143, 0.9483993
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7721679, 0.7716608
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9215994, 0.9241414
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6752725, 0.6740550
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5491925, 0.5471041
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6955948, 0.6917230

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2962764, upper bound: 0.2964840
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2941582, upper bound: 0.2941934
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0491228, 1.0501099
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7857844, 0.7850528
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7926629, 0.7912621
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8928239, 0.9002762
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9478889, 0.9454241
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7714355, 0.7723929
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9226851, 0.9230552
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6731656, 0.6761620
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5487423, 0.5475538
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6934342, 0.6938828

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941723, upper bound: 0.2962983
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964619, upper bound: 0.2962984
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0503507, 1.0488822
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7836555, 0.7871823
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7929420, 0.7909844
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8987103, 0.8943895
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9439559, 0.9493580
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7714660, 0.7723629
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9227693, 0.9229715
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6760261, 0.6733013
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5488906, 0.5474060
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6951818, 0.6921360

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941798, upper bound: 0.2985806
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941582, upper bound: 0.2962895
time: 6.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0501099, 1.0491230
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7850527, 0.7857845
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7912621, 0.7926626
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9002764, 0.8928239
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9454241, 0.9478886
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7723930, 0.7714355
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9230552, 0.9226854
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6761620, 0.6731654
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5475539, 0.5487423
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6938829, 0.6934341

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2962989, upper bound: 0.2964613
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2941582, upper bound: 0.2941723
time: 4.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0484266, 1.0508060
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7862219, 0.7846160
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7914314, 0.7924948
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8944359, 0.8986639
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9483991, 0.9449143
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7716610, 0.7721679
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9241414, 0.9215995
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6740551, 0.6752725
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5471041, 0.5491925
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6917229, 0.6955948

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2941928, upper bound: 0.2962744
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964642, upper bound: 0.2962786
time: 4.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 1, lower bound: -0.2941571, upper bound: 0.2986031
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 1, lower bound: -0.2941582, upper bound: 0.2963125
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 1, lower bound: -0.2962764, upper bound: 0.2964840
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.59
Output dim: 1, lower bound: -0.2941582, upper bound: 0.2941934
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 1, lower bound: -0.2941723, upper bound: 0.2962983
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 1, lower bound: -0.2964619, upper bound: 0.2962984
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 1, lower bound: -0.2941798, upper bound: 0.2985806
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 1, lower bound: -0.2941582, upper bound: 0.2962895
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 1, lower bound: -0.2962989, upper bound: 0.2964613
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.59
Output dim: 1, lower bound: -0.2941582, upper bound: 0.2941723
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 1, lower bound: -0.2941928, upper bound: 0.2962744
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 1, lower bound: -0.2964642, upper bound: 0.2962786

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

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.37 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2881014, upper bound: 0.2957660
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2913198, upper bound: 0.2925459
time: 3.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2881027, upper bound: 0.2934725
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2913163, upper bound: 0.2902550
time: 4.41 seconds

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

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2902208, upper bound: 0.2936457
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2934382, upper bound: 0.2904272
time: 4.24 seconds

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

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2881144, upper bound: 0.2934582
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2913305, upper bound: 0.2902410
time: 3.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2904077, upper bound: 0.2934592
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2936247, upper bound: 0.2902425
time: 4.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2881238, upper bound: 0.2957436
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2913397, upper bound: 0.2925238
time: 4.02 seconds

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

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2881027, upper bound: 0.2934535
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2913385, upper bound: 0.2902331
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2902429, upper bound: 0.2936236
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2934601, upper bound: 0.2904050
time: 4.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2881368, upper bound: 0.2934380
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2913527, upper bound: 0.2902189
time: 4.04 seconds

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

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2904077, upper bound: 0.2934367
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2936467, upper bound: 0.2902206
time: 4.09 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2881014, upper bound: 0.2957660
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2913198, upper bound: 0.2925459
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2881027, upper bound: 0.2934725
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2913163, upper bound: 0.2902550
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2902208, upper bound: 0.2936457
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2934382, upper bound: 0.2904272
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2881144, upper bound: 0.2934582
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2913305, upper bound: 0.2902410
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2904077, upper bound: 0.2934592
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2936247, upper bound: 0.2902425
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2881238, upper bound: 0.2957436
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2913397, upper bound: 0.2925238
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2881027, upper bound: 0.2934535
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2913385, upper bound: 0.2902331
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2902429, upper bound: 0.2936236
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2934601, upper bound: 0.2904050
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2881368, upper bound: 0.2934380
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2913527, upper bound: 0.2902189
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2904077, upper bound: 0.2934367
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.28
Output dim: 1, lower bound: -0.2936467, upper bound: 0.2902206

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0399313, 1.0317142
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7799735, 0.7849566
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7854779, 0.7830384
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8907104, 0.8810596
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9278340, 0.9386764
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7638452, 0.7597151
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9183662, 0.9178725
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6687502, 0.6684785
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5443008, 0.5386071
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6949058, 0.6859628

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2615

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2658708, upper bound: 0.2696492
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2658708, upper bound: 0.2696492
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.0392346, 1.0324106
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.7804112, 0.7845198
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.7842467, 0.7842712
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.8923223, 0.8794472
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9283447, 0.9381664
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.7640705, 0.7594900
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -0.9198225, 0.9164170
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.6696395, 0.6675889
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.5426625, 0.5402458
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.6931944, 0.6876746

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2615

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2658918, upper bound: 0.2696272
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2658918, upper bound: 0.2696272
time: 4.29 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 23.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.20
Output dim: 1, lower bound: -0.2658708, upper bound: 0.2696492
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.20
Output dim: 1, lower bound: -0.2658708, upper bound: 0.2696492
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.20
Output dim: 1, lower bound: -0.2658918, upper bound: 0.2696272
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.20
Output dim: 1, lower bound: -0.2658918, upper bound: 0.2696272
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=0.8209632635116577
rel_dist={1: [-0.29885889706970215, 0.2988555407171436]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6220
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6220

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4133354, upper bound: 0.4134234
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4133377, upper bound: 0.4133345
time: 6.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.20 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.20
Output dim: 1, lower bound: -0.4133354, upper bound: 0.4134234
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.20
Output dim: 1, lower bound: -0.4133377, upper bound: 0.4133345

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1551342, 1.1537414
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8754073, 0.8762823
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8752794, 0.8728142
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0295815, 1.0328053
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0508687, 1.0518899
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8586926, 0.8591433
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0248077, 1.0277199
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7581999, 0.7599788
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6437044, 0.6404269
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8083534, 0.8049296

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4085397, upper bound: 0.4134225
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4085421, upper bound: 0.4086310
time: 5.92 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1537409, 1.1551342
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8762823, 0.8754073
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8728142, 0.8752794
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.0328054, 1.0295817
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0518901, 1.0508685
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8591433, 0.8586924
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0277200, 1.0248077
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7599788, 0.7581998
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6404269, 0.6437043
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.8049295, 0.8083534

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4085421, upper bound: 0.4133342
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4134209, upper bound: 0.4085384
time: 4.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.66 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.66
Output dim: 1, lower bound: -0.4085397, upper bound: 0.4134225
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.66
Output dim: 1, lower bound: -0.4085421, upper bound: 0.4086310
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.66
Output dim: 1, lower bound: -0.4085421, upper bound: 0.4133342
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.66
Output dim: 1, lower bound: -0.4134209, upper bound: 0.4085384

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1381817, 1.1329408
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8476337, 0.8536415
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8619647, 0.8564782
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9919305, 0.9866054
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0028379, 1.0127463
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8565867, 0.8574274
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0099723, 1.0156283
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7408264, 0.7386632
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6279397, 0.6210891
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7777872, 0.7674459

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4043046, upper bound: 0.4134154
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4085355, upper bound: 0.4091844
time: 4.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1343336, 1.1367884
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8527666, 0.8485086
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8589435, 0.8594992
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9833817, 0.9951541
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0117247, 1.0038590
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8569765, 0.8570374
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0127163, 1.0128845
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7368844, 0.7426053
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6243668, 0.6246623
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7708697, 0.7743634

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4090966, upper bound: 0.4086266
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4133276, upper bound: 0.4043880
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1367884, 1.1343336
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8485087, 0.8527665
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8594992, 0.8589436
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9951541, 0.9833817
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0038593, 1.0117249
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8570375, 0.8569765
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0128846, 1.0127163
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7426054, 0.7368842
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6246623, 0.6243666
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7743635, 0.7708697

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4043888, upper bound: 0.4133263
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4086246, upper bound: 0.4090952
time: 4.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1329408, 1.1381814
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8536416, 0.8476336
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8564782, 0.8619646
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9866054, 0.9919305
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0127461, 1.0028377
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8574274, 0.8565867
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0156283, 1.0099723
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7386634, 0.7408264
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6210892, 0.6279397
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7674460, 0.7777872

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5829
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5829

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4091823, upper bound: 0.4085341
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4134166, upper bound: 0.4043041
time: 4.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.90 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.4043046, upper bound: 0.4134154
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.4085355, upper bound: 0.4091844
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.4090966, upper bound: 0.4086266
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.4133276, upper bound: 0.4043880
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.4043888, upper bound: 0.4133263
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.4086246, upper bound: 0.4090952
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.4091823, upper bound: 0.4085341
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.90
Output dim: 1, lower bound: -0.4134166, upper bound: 0.4043041

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1375370, 1.1318147
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8410465, 0.8498490
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8573902, 0.8485470
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9845479, 0.9823549
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9959121, 1.0087576
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8522067, 0.8549011
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0086238, 1.0148513
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7401853, 0.7382941
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6243105, 0.6147873
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7742662, 0.7613289

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4040419, upper bound: 0.4129747
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4040397, upper bound: 0.4083949
time: 3.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1370554, 1.1322966
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8438424, 0.8470544
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8540335, 0.8519064
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9876790, 0.9792229
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9988508, 1.0058205
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8540614, 0.8530471
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0091963, 1.0142800
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7404571, 0.7380222
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6216379, 0.6174608
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7716703, 0.7639266

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4082832, upper bound: 0.4087348
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4040397, upper bound: 0.4041581
time: 6.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1336889, 1.1356626
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8461794, 0.8447161
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8543692, 0.8515680
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9759991, 0.9909036
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0047994, 0.9998703
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8525965, 0.8545113
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0113678, 1.0121074
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7362432, 0.7422361
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6207374, 0.6183604
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7673488, 0.7682464

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4040656, upper bound: 0.4083669
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4086484, upper bound: 0.4083691
time: 3.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1332078, 1.1361444
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8489753, 0.8419214
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8510122, 0.8549275
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9791305, 0.9877716
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0077376, 0.9969335
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8544514, 0.8526573
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0119400, 1.0115361
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7365150, 0.7419643
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6180649, 0.6210339
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7647529, 0.7708441

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4083092, upper bound: 0.4041267
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4128885, upper bound: 0.4041322
time: 4.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1361446, 1.1332078
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8419214, 0.8489753
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8549275, 0.8510122
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9877717, 0.9791303
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9969335, 1.0077379
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8526576, 0.8544512
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0115361, 1.0119401
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7419643, 0.7365150
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6210340, 0.6180649
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7708440, 0.7647527

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4041298, upper bound: 0.4128875
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4040397, upper bound: 0.4083085
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1356626, 1.1336892
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8447160, 0.8461794
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8515680, 0.8543689
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9909036, 0.9759991
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9998703, 1.0047991
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8545115, 0.8525964
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0121074, 1.0113679
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7422361, 0.7362432
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6183604, 0.6207374
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7682467, 0.7673488

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4083698, upper bound: 0.4086455
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4083675, upper bound: 0.4040651
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1322966, 1.1370556
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8470544, 0.8438424
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8519065, 0.8540332
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9792230, 0.9876790
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0058208, 0.9988508
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8530474, 0.8540614
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0142798, 1.0091963
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7380223, 0.7404571
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6174610, 0.6216379
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7639265, 0.7716702

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4041559, upper bound: 0.4082801
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4087362, upper bound: 0.4082824
time: 4.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1318150, 1.1375370
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8498491, 0.8410465
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8485470, 0.8573900
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9823549, 0.9845479
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0087576, 0.9959121
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8549011, 0.8522066
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0148511, 1.0086241
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7382941, 0.7401853
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6147875, 0.6243104
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7613292, 0.7742662

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4083958, upper bound: 0.4040368
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4129754, upper bound: 0.4040391
time: 3.88 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4040419, upper bound: 0.4129747
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4040397, upper bound: 0.4083949
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4082832, upper bound: 0.4087348
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4040397, upper bound: 0.4041581
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4040656, upper bound: 0.4083669
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4086484, upper bound: 0.4083691
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4083092, upper bound: 0.4041267
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4128885, upper bound: 0.4041322
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4041298, upper bound: 0.4128875
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4040397, upper bound: 0.4083085
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4083698, upper bound: 0.4086455
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4083675, upper bound: 0.4040651
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4041559, upper bound: 0.4082801
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4087362, upper bound: 0.4082824
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4083958, upper bound: 0.4040368
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.40
Output dim: 1, lower bound: -0.4129754, upper bound: 0.4040391

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1353407, 1.1284313
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8406677, 0.8492663
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8555906, 0.8457806
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9838047, 0.9812118
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9901567, 1.0050206
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8517208, 0.8541585
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0077391, 1.0134894
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7392271, 0.7376704
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6233871, 0.6133689
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7728109, 0.7590911

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3949325, upper bound: 0.4086498
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3996130, upper bound: 0.4038718
time: 3.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1341538, 1.1296191
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8404636, 0.8490632
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8546238, 0.8467476
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9834049, 0.9816116
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9921746, 1.0030026
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8514640, 0.8538992
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0072620, 1.0130117
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7388918, 0.7373359
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6228919, 0.6138607
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7720284, 0.7598637

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3949303, upper bound: 0.4040600
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3996108, upper bound: 0.3993000
time: 4.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1348591, 1.1289129
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8434639, 0.8464717
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8522339, 0.8491402
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9869359, 0.9780798
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9930954, 1.0020835
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8535755, 0.8523047
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0083113, 1.0129180
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7394989, 0.7373986
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6207145, 0.6160424
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7702148, 0.7616888

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3991838, upper bound: 0.4043857
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4039026, upper bound: 0.3996241
time: 3.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1336722, 1.1301007
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8432598, 0.8462684
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8512671, 0.8501071
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9865360, 0.9784796
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9951134, 1.0000653
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8533189, 0.8520453
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0078344, 1.0124404
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7391636, 0.7370641
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6202193, 0.6165344
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7694325, 0.7624614

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3991816, upper bound: 0.3997982
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4039003, upper bound: 0.3950529
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1314936, 1.1322792
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8453937, 0.8441334
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8525698, 0.8488016
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9752560, 0.9897605
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9990439, 0.9961333
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8515947, 0.8537687
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0095284, 1.0107455
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7352850, 0.7409427
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6198108, 0.6169418
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7658834, 0.7660086

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3949585, upper bound: 0.4040322
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3996390, upper bound: 0.3992719
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1303058, 1.1334662
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8455968, 0.8443373
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8516028, 0.8497684
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9748561, 0.9901603
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0010619, 0.9941154
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8518538, 0.8540254
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0100060, 1.0112226
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7356195, 0.7412779
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6193188, 0.6174370
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7651110, 0.7667911

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3995294, upper bound: 0.4040353
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4042291, upper bound: 0.3992743
time: 3.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1310120, 1.1327610
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8481894, 0.8413388
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8492129, 0.8521612
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9783871, 0.9866285
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0019827, 0.9931960
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8534493, 0.8519148
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0101006, 1.0101742
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7355568, 0.7406709
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6171383, 0.6196154
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7632875, 0.7686063

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3992098, upper bound: 0.3997673
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4039286, upper bound: 0.3950250
time: 4.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1298242, 1.1339478
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8483925, 0.8415427
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8482461, 0.8531280
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9779873, 0.9870282
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0040011, 0.9911783
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8537087, 0.8521715
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0105782, 1.0106514
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7358913, 0.7410061
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6166464, 0.6201106
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7625151, 0.7693888

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4037815, upper bound: 0.3997729
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4085186, upper bound: 0.3950269
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1339478, 1.1298242
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8415427, 0.8483926
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8531280, 0.8482461
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9870284, 0.9779872
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9911780, 1.0040011
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8521717, 0.8537086
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0106514, 1.0105782
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7410061, 0.7358913
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6201106, 0.6166463
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7693887, 0.7625149

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3950276, upper bound: 0.4085176
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3997705, upper bound: 0.4037805
time: 6.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1327610, 1.1310120
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8413386, 0.8481894
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8521612, 0.8492129
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9866288, 0.9783869
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9931960, 1.0019827
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8519149, 0.8534493
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0101743, 1.0101006
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7406709, 0.7355568
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6196154, 0.6171383
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7686062, 0.7632875

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3949327, upper bound: 0.4039284
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3997682, upper bound: 0.3992088
time: 4.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1334662, 1.1303058
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8443375, 0.8455967
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8497684, 0.8516028
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9901605, 0.9748560
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9941154, 1.0010624
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8540256, 0.8518538
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0112226, 1.0100060
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7412779, 0.7356195
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6174370, 0.6193188
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7667911, 0.7651110

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3992749, upper bound: 0.4042278
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4040353, upper bound: 0.3995318
time: 5.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1322789, 1.1314936
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8441334, 0.8453934
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8488019, 0.8525696
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9897606, 0.9752558
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9961333, 0.9990442
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8537688, 0.8515946
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0107455, 1.0095284
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7409427, 0.7352850
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6169419, 0.6198108
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7660086, 0.7658836

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3949327, upper bound: 0.3996383
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4040330, upper bound: 0.3949584
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1301007, 1.1336722
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8462687, 0.8432597
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8501072, 0.8512671
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9784796, 0.9865358
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0000653, 0.9951134
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8520453, 0.8533188
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0124404, 1.0078343
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7370641, 0.7391636
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6165344, 0.6202193
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7624614, 0.7694324

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3950536, upper bound: 0.4038994
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3997965, upper bound: 0.3991807
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1289129, 1.1348591
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8464718, 0.8434637
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8491404, 0.8522338
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9780800, 0.9869356
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0020833, 0.9930956
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8523047, 0.8535755
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0129180, 1.0083114
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7373986, 0.7394990
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6160424, 0.6207145
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7616887, 0.7702148

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3996250, upper bound: 0.4039017
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4043865, upper bound: 0.3991828
time: 4.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1296191, 1.1341538
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8490629, 0.8404638
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8467476, 0.8546238
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9816117, 0.9834048
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0030026, 0.9921746
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8538992, 0.8514640
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0130117, 1.0072620
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7373359, 0.7388918
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6138607, 0.6228919
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7598639, 0.7720284

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3993009, upper bound: 0.3996100
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4040613, upper bound: 0.3949326
time: 4.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1284313, 1.1353407
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8492665, 0.8406677
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8457806, 0.8555906
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9812119, 0.9838046
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.0050206, 0.9901569
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8541586, 0.8517208
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0134892, 1.0077392
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7376704, 0.7392272
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6133689, 0.6233871
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7590911, 0.7728109

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4038726, upper bound: 0.3996117
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4086511, upper bound: 0.3949317
time: 4.23 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3949325, upper bound: 0.4086498
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3996130, upper bound: 0.4038718
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3949303, upper bound: 0.4040600
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3996108, upper bound: 0.3993000
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3991838, upper bound: 0.4043857
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.4039026, upper bound: 0.3996241
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3991816, upper bound: 0.3997982
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.4039003, upper bound: 0.3950529
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3949585, upper bound: 0.4040322
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3996390, upper bound: 0.3992719
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3995294, upper bound: 0.4040353
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.4042291, upper bound: 0.3992743
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3992098, upper bound: 0.3997673
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.4039286, upper bound: 0.3950250
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.4037815, upper bound: 0.3997729
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.4085186, upper bound: 0.3950269
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3950276, upper bound: 0.4085176
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3997705, upper bound: 0.4037805
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3949327, upper bound: 0.4039284
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3997682, upper bound: 0.3992088
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3992749, upper bound: 0.4042278
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.4040353, upper bound: 0.3995318
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3949327, upper bound: 0.3996383
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.4040330, upper bound: 0.3949584
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3950536, upper bound: 0.4038994
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3997965, upper bound: 0.3991807
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3996250, upper bound: 0.4039017
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.4043865, upper bound: 0.3991828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.3993009, upper bound: 0.3996100
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.4040613, upper bound: 0.3949326
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.4038726, upper bound: 0.3996117
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.82
Output dim: 1, lower bound: -0.4086511, upper bound: 0.3949317

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1270146, 1.1105807
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8379040, 0.8478700
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8491781, 0.8442991
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9867144, 0.9674127
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9803009, 1.0019855
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8502886, 0.8420283
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0092843, 1.0082968
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7332920, 0.7327487
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6183300, 0.6069426
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7747531, 0.7568675

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1403
type: RSZ, layer: 3, pos: 767
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1774
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2489
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 723
type: RSZ, layer: 3, pos: 1776
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1964
type: RSZ, layer: 3, pos: 740
type: RSZ, layer: 3, pos: 1164

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2615

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3419152, upper bound: 0.3489311
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3419152, upper bound: 0.3489311
time: 4.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.1353407, 1.1201055
1: 2.4124451, 3.9131384, 2.4124451, 3.9131384, -0.8406677, 0.8465027
2: -6.5181541, -4.9777999, -6.5181541, -4.9777999, -0.8555906, 0.8393683
3: -11.4270353, -9.5017948, -11.4270353, -9.5017948, -0.9700055, 0.9812118
4: -4.3657799, -2.8065395, -4.3657799, -2.8065395, -0.9901567, 0.9951646
5: -12.3431225, -10.5792007, -12.3431225, -10.5792007, -0.8395905, 0.8541585
6: -10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.0025465, 1.0134894
7: -4.2142544, -2.6923499, -4.2142544, -2.6923499, -0.7343057, 0.7376704
8: -3.2913580, -1.8388863, -3.2913580, -1.8388863, -0.6233871, 0.6083118
9: -12.0051117, -10.4397650, -12.0051117, -10.4397650, -0.7705870, 0.7590911

Time for backsubstitution: 14.41 seconds
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=0.8787916898727417
rel_dist={1: [-0.4134268291479084, 0.41342682172075307]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 2286.54 seconds
