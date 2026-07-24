## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.5776130088


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6158381, 1.6158381)
1: (-13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2368627, 1.2368627)
2: (-10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.3015485, 1.3015490)
3: (-14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3128452, 1.3128452)
4: (7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4061766, 1.4061770)
5: (-7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2984648, 1.2984648)
6: (-9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2679276, 1.2679276)
7: (-5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2577901, 1.2577901)
8: (-4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4829001, 1.4829001)
9: (-7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3555326, 1.3555326)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.68 + 38.02 = 60.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.5781902, upper bound: 0.5781905

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4599
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 4599

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781879, upper bound: 0.5781384
time: 5.20 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781372, upper bound: 0.5781891
time: 5.49 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.91 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.91
Output dim: 4, lower bound: -0.5781879, upper bound: 0.5781384
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.91
Output dim: 4, lower bound: -0.5781372, upper bound: 0.5781891

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6175032, 1.6148224
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2417393, 1.2338347
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2969298, 1.3089676
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3114681, 1.3150830
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4066896, 1.4058762
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2985916, 1.2983851
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2679043, 1.2679620
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2578416, 1.2577558
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4824138, 1.4836984
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3556280, 1.3554764

Time for backsubstitution: 20.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 6236

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763673, upper bound: 0.5781348
time: 11.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781855, upper bound: 0.5763204
time: 4.88 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6148224, 1.6158381
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2338347, 1.2368627
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.3015485, 1.2969298
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3128452, 1.3114681
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4058762, 1.4061770
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2983847, 1.2984648
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2679276, 1.2679038
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2577558, 1.2577901
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4829001, 1.4824138
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3554754, 1.3555326

Time for backsubstitution: 21.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 6236

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763193, upper bound: 0.5781868
time: 5.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781348, upper bound: 0.5763683
time: 5.81 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 33.56 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.56
Output dim: 4, lower bound: -0.5763673, upper bound: 0.5781348
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 33.56
Output dim: 4, lower bound: -0.5781855, upper bound: 0.5763204
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.56
Output dim: 4, lower bound: -0.5763193, upper bound: 0.5781868
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 33.56
Output dim: 4, lower bound: -0.5781348, upper bound: 0.5763683

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6006918, 1.6013498
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2450938, 1.2364092
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2742405, 1.2904630
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3148818, 1.3175559
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4024806, 1.4043674
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2976508, 1.3003592
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2504435, 1.2470217
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2498841, 1.2511220
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4224658, 1.4096251
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3542728, 1.3538494

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5763670, upper bound: 0.5774271
time: 5.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5756456, upper bound: 0.5781344
time: 6.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6040306, 1.5980110
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2443142, 1.2371888
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2784252, 1.2862782
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3139415, 1.3184967
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4051814, 1.4016671
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.3005662, 1.2974439
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2469640, 1.2505016
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2512078, 1.2497983
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4083400, 1.4237494
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3540010, 1.3541203

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781853, upper bound: 0.5756028
time: 4.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5774725, upper bound: 0.5763202
time: 5.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.5980110, 1.6023684
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2371888, 1.2394366
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2788596, 1.2784252
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3162599, 1.3139410
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4016671, 1.4046693
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2974439, 1.3004389
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2504678, 1.2469640
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2497983, 1.2511559
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4229522, 1.4083405
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3541212, 1.3539047

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5763191, upper bound: 0.5774726
time: 7.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5756018, upper bound: 0.5781855
time: 7.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6013498, 1.5990295
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2364092, 1.2402163
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2830443, 1.2742405
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3153186, 1.3148823
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4043670, 1.4019690
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.3003592, 1.2975235
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2469883, 1.2504435
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2511220, 1.2498322
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4088273, 1.4224648
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3538485, 1.3541756

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781346, upper bound: 0.5756455
time: 5.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5774260, upper bound: 0.5763681
time: 5.26 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 33.20 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 33.20
Output dim: 4, lower bound: -0.5763670, upper bound: 0.5774271
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.20
Output dim: 4, lower bound: -0.5756456, upper bound: 0.5781344
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.20
Output dim: 4, lower bound: -0.5781853, upper bound: 0.5756028
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 33.20
Output dim: 4, lower bound: -0.5774725, upper bound: 0.5763202
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 33.20
Output dim: 4, lower bound: -0.5763191, upper bound: 0.5774726
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.20
Output dim: 4, lower bound: -0.5756018, upper bound: 0.5781855
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.20
Output dim: 4, lower bound: -0.5781346, upper bound: 0.5756455
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 33.20
Output dim: 4, lower bound: -0.5774260, upper bound: 0.5763681

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6023083, 1.6026373
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2468505, 1.2378082
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2724781, 1.2883472
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3174973, 1.3196383
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4024782, 1.4044294
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2994752, 1.3018126
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2484341, 1.2445035
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2500186, 1.2513027
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4205627, 1.4073415
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3519864, 1.3519430

Time for backsubstitution: 21.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 116

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5740988, upper bound: 0.5781167
time: 4.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5756273, upper bound: 0.5765860
time: 6.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6053181, 1.5996265
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2457132, 1.2389455
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2763095, 1.2845159
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3160229, 1.3211122
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4052429, 1.4016647
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.3020196, 1.2992673
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2444453, 1.2484918
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2513885, 1.2499328
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4060574, 1.4218469
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3520951, 1.3518343

Time for backsubstitution: 22.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 116

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5766369, upper bound: 0.5755842
time: 5.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781660, upper bound: 0.5740555
time: 4.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.5996265, 1.6036539
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2389455, 1.2408361
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2770967, 1.2763100
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3188744, 1.3160238
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4016647, 1.4047322
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2992673, 1.3018913
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2484570, 1.2444453
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2499328, 1.2513361
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4210491, 1.4060569
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3518348, 1.3519983

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 116

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5740543, upper bound: 0.5781672
time: 6.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5755830, upper bound: 0.5766370
time: 6.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6026373, 1.6006432
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2378082, 1.2419734
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2809286, 1.2724781
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3174000, 1.3174977
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4044294, 1.4019675
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.3018117, 1.2993460
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2444687, 1.2484341
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2513027, 1.2499661
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4065437, 1.4205627
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3519425, 1.3518906

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 116

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5765861, upper bound: 0.5756285
time: 5.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781155, upper bound: 0.5741000
time: 5.92 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 33.68 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 4, lower bound: -0.5740988, upper bound: 0.5781167
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 4, lower bound: -0.5756273, upper bound: 0.5765860
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 4, lower bound: -0.5766369, upper bound: 0.5755842
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 4, lower bound: -0.5781660, upper bound: 0.5740555
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 4, lower bound: -0.5740543, upper bound: 0.5781672
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 4, lower bound: -0.5755830, upper bound: 0.5766370
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 4, lower bound: -0.5765861, upper bound: 0.5756285
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 4, lower bound: -0.5781155, upper bound: 0.5741000

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6022940, 1.6026196
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2468615, 1.2378178
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2724688, 1.2883363
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3175111, 1.3196545
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4024739, 1.4044256
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2994699, 1.3018093
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2484303, 1.2444992
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2500415, 1.2513304
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4205909, 1.4073653
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3519759, 1.3519344

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 220
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1444
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 2243
type: DSZ, layer: 3, pos: 764
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 715
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1695
type: DSZ, layer: 3, pos: 1496
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 218
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 568
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1144
type: DSZ, layer: 3, pos: 1149
type: DSZ, layer: 3, pos: 2006
type: DSZ, layer: 3, pos: 1114
type: DSZ, layer: 3, pos: 2619
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 2486
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 3, pos: 1438

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5586036, upper bound: 0.5578058
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5539109, upper bound: 0.5625696
time: 4.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6053019, 1.5996127
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2457228, 1.2389569
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2762983, 1.2845063
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3160396, 1.3211255
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4052396, 1.4016600
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.3020163, 1.2992635
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2444415, 1.2484884
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2514162, 1.2499557
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4060817, 1.4218760
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3520865, 1.3518238

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 220
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1444
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 2243
type: DSZ, layer: 3, pos: 764
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 715
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1695
type: DSZ, layer: 3, pos: 1496
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 218
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 568
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1144
type: DSZ, layer: 3, pos: 1149
type: DSZ, layer: 3, pos: 2006
type: DSZ, layer: 3, pos: 1114
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 2619
type: DSZ, layer: 3, pos: 2486
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 3, pos: 1438

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5626269, upper bound: 0.5538621
time: 4.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5578569, upper bound: 0.5585450
time: 6.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.5996122, 1.6036382
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2389569, 1.2408452
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2770875, 1.2762985
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3188863, 1.3160400
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4016595, 1.4047279
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2992630, 1.3018885
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2484546, 1.2444415
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2499557, 1.2513642
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4210773, 1.4060807
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3518233, 1.3519897

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 220
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1444
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 2243
type: DSZ, layer: 3, pos: 764
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 715
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1695
type: DSZ, layer: 3, pos: 1496
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 218
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 568
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1144
type: DSZ, layer: 3, pos: 1149
type: DSZ, layer: 3, pos: 2006
type: DSZ, layer: 3, pos: 1114
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 2619
type: DSZ, layer: 3, pos: 2486
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 1438

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5585453, upper bound: 0.5578569
time: 7.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5538608, upper bound: 0.5626270
time: 6.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6026201, 1.6006303
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2378178, 1.2419844
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2809169, 1.2724686
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3174157, 1.3175111
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4044251, 1.4019623
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.3018093, 1.2993426
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2444654, 1.2484303
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2513304, 1.2499890
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4065681, 1.4205914
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3519340, 1.3518791

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 220
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1444
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 2243
type: DSZ, layer: 3, pos: 764
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 715
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1695
type: DSZ, layer: 3, pos: 1496
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 218
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 568
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1144
type: DSZ, layer: 3, pos: 1149
type: DSZ, layer: 3, pos: 2006
type: DSZ, layer: 3, pos: 1114
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 2619
type: DSZ, layer: 3, pos: 2486
type: DSZ, layer: 3, pos: 2494

Time for candidate selection: 0.47 seconds

### Candidate
type: DSZ, layer: 3, pos: 1438

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5625683, upper bound: 0.5539121
time: 5.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5578046, upper bound: 0.5586036
time: 6.99 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 34.90 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 34.90
Output dim: 4, lower bound: -0.5586036, upper bound: 0.5578058
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 34.90
Output dim: 4, lower bound: -0.5539109, upper bound: 0.5625696
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 34.90
Output dim: 4, lower bound: -0.5626269, upper bound: 0.5538621
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 34.90
Output dim: 4, lower bound: -0.5578569, upper bound: 0.5585450
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 34.90
Output dim: 4, lower bound: -0.5585453, upper bound: 0.5578569
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 34.90
Output dim: 4, lower bound: -0.5538608, upper bound: 0.5626270
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 34.90
Output dim: 4, lower bound: -0.5625683, upper bound: 0.5539121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 34.90
Output dim: 4, lower bound: -0.5578046, upper bound: 0.5586036

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 60.70 + 490.15 = 550.85 seconds
