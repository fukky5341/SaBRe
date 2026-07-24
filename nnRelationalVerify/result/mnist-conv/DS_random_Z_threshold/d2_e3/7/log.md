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
execution time: IAR + RelationalAnalysis = 23.95 + 37.09 = 61.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.5781902, upper bound: 0.5781905

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6236

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5781889
time: 9.99 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781880, upper bound: 0.5763703
time: 4.66 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 14.66 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 14.66
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5781889
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 14.66
Output dim: 4, lower bound: -0.5781880, upper bound: 0.5763703

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.5990295, 1.6023684
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2402163, 1.2394366
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2788596, 1.2830439
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3162599, 1.3153181
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4019685, 1.4046693
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2975240, 1.3004389
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2504678, 1.2469883
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2498322, 1.2511559
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4229522, 1.4088278
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3541756, 1.3539047

Time for backsubstitution: 23.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 4599
type: DSZ, layer: 1, pos: 871

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 116

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5748347, upper bound: 0.5781685
time: 7.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5763507, upper bound: 0.5766405
time: 5.08 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6023684, 1.5990295
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2394366, 1.2402163
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2830443, 1.2788596
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3153186, 1.3162594
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4046693, 1.4019690
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.3004394, 1.2975235
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2469883, 1.2504678
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2511559, 1.2498322
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4088273, 1.4229522
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3539047, 1.3541756

Time for backsubstitution: 23.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4599
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4599

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781855, upper bound: 0.5763204
time: 4.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781348, upper bound: 0.5763683
time: 5.55 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 33.42 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.42
Output dim: 4, lower bound: -0.5748347, upper bound: 0.5781685
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 33.42
Output dim: 4, lower bound: -0.5763507, upper bound: 0.5766405
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.42
Output dim: 4, lower bound: -0.5781855, upper bound: 0.5763204
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 33.42
Output dim: 4, lower bound: -0.5781348, upper bound: 0.5763683

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.5990143, 1.6023502
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2402272, 1.2394457
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2788498, 1.2830324
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3162723, 1.3153338
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4019632, 1.4046645
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2975197, 1.3004355
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2504640, 1.2469835
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2498560, 1.2511845
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4229803, 1.4088507
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3541656, 1.3538957

Time for backsubstitution: 23.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4599
type: DSZ, layer: 1, pos: 871

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4599

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5748327, upper bound: 0.5781170
time: 6.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5747854, upper bound: 0.5781675
time: 5.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1

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

Time for backsubstitution: 23.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781853, upper bound: 0.5756028
time: 4.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5774725, upper bound: 0.5763202
time: 5.38 seconds

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

Time for backsubstitution: 23.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 871

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 116

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5765863, upper bound: 0.5763499
time: 5.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781157, upper bound: 0.5748331
time: 7.99 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 37.30 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 37.30
Output dim: 4, lower bound: -0.5748327, upper bound: 0.5781170
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 37.30
Output dim: 4, lower bound: -0.5747854, upper bound: 0.5781675
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 37.30
Output dim: 4, lower bound: -0.5781853, upper bound: 0.5756028
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 37.30
Output dim: 4, lower bound: -0.5774725, upper bound: 0.5763202
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 37.30
Output dim: 4, lower bound: -0.5765863, upper bound: 0.5763499
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 37.30
Output dim: 4, lower bound: -0.5781157, upper bound: 0.5748331

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6006784, 1.6013346
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2451043, 1.2364182
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2742312, 1.2904515
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3148971, 1.3175726
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4024744, 1.4043617
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2976451, 1.3003554
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2504406, 1.2470188
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2499080, 1.2511511
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4224939, 1.4096489
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3542614, 1.3538394

Time for backsubstitution: 23.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5748325, upper bound: 0.5774100
time: 4.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5740988, upper bound: 0.5781167
time: 5.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.5979977, 1.6023502
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2371998, 1.2394457
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2788498, 1.2784138
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3162723, 1.3139582
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4016609, 1.4046645
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2974391, 1.3004355
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2504640, 1.2469606
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2498221, 1.2511845
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4229803, 1.4083648
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3541088, 1.3538957

Time for backsubstitution: 23.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5747852, upper bound: 0.5774566
time: 7.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5740543, upper bound: 0.5781672
time: 6.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 23.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 116

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 116

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5766369, upper bound: 0.5755842
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781660, upper bound: 0.5740555
time: 4.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6013336, 1.5990143
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2364182, 1.2402272
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2830322, 1.2742314
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3153338, 1.3148966
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4043617, 1.4019632
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.3003554, 1.2975187
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2469835, 1.2504406
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2511511, 1.2498560
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4088507, 1.4224939
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3538389, 1.3541656

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781155, upper bound: 0.5741000
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5774088, upper bound: 0.5748336
time: 5.50 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 33.97 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.97
Output dim: 4, lower bound: -0.5748325, upper bound: 0.5774100
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.97
Output dim: 4, lower bound: -0.5740988, upper bound: 0.5781167
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.97
Output dim: 4, lower bound: -0.5747852, upper bound: 0.5774566
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.97
Output dim: 4, lower bound: -0.5740543, upper bound: 0.5781672
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.97
Output dim: 4, lower bound: -0.5766369, upper bound: 0.5755842
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.97
Output dim: 4, lower bound: -0.5781660, upper bound: 0.5740555
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.97
Output dim: 4, lower bound: -0.5781155, upper bound: 0.5741000
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.97
Output dim: 4, lower bound: -0.5774088, upper bound: 0.5748336

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 22.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 220
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 2619
type: DSZ, layer: 3, pos: 2243
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 568
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 764
type: DSZ, layer: 3, pos: 715
type: DSZ, layer: 3, pos: 2486
type: DSZ, layer: 3, pos: 2006
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1496
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1149
type: DSZ, layer: 3, pos: 1444
type: DSZ, layer: 3, pos: 218
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1144
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1695
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 1114
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 571

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5720791, upper bound: 0.5772234
time: 5.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5731905, upper bound: 0.5760117
time: 5.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 22.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2006
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1114
type: DSZ, layer: 3, pos: 1695
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 715
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 764
type: DSZ, layer: 3, pos: 1144
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 220
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2243
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 218
type: DSZ, layer: 3, pos: 1149
type: DSZ, layer: 3, pos: 1444
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 2486
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 568
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 1496
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2619
type: DSZ, layer: 3, pos: 2487

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2006

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5720573, upper bound: 0.5761829
time: 5.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5720477, upper bound: 0.5761905
time: 5.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 218
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2243
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 715
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1496
type: DSZ, layer: 3, pos: 220
type: DSZ, layer: 3, pos: 764
type: DSZ, layer: 3, pos: 2619
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1114
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1695
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 2006
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 568
type: DSZ, layer: 3, pos: 1444
type: DSZ, layer: 3, pos: 2486
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1149
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1144
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 3109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 218

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5780775, upper bound: 0.5732181
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5766164, upper bound: 0.5739763
time: 5.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 23.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2486
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 764
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 2006
type: DSZ, layer: 3, pos: 1695
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 2243
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 1496
type: DSZ, layer: 3, pos: 2619
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 568
type: DSZ, layer: 3, pos: 715
type: DSZ, layer: 3, pos: 218
type: DSZ, layer: 3, pos: 1144
type: DSZ, layer: 3, pos: 1114
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1444
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 220
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 1149
type: DSZ, layer: 3, pos: 2832

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2486

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5778742, upper bound: 0.5728347
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5766933, upper bound: 0.5739728
time: 5.44 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 33.55 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 33.55
Output dim: 4, lower bound: -0.5720791, upper bound: 0.5772234
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 33.55
Output dim: 4, lower bound: -0.5731905, upper bound: 0.5760117
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 33.55
Output dim: 4, lower bound: -0.5720573, upper bound: 0.5761829
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 33.55
Output dim: 4, lower bound: -0.5720477, upper bound: 0.5761905
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 33.55
Output dim: 4, lower bound: -0.5780775, upper bound: 0.5732181
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 33.55
Output dim: 4, lower bound: -0.5766164, upper bound: 0.5739763
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 33.55
Output dim: 4, lower bound: -0.5778742, upper bound: 0.5728347
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 33.55
Output dim: 4, lower bound: -0.5766933, upper bound: 0.5739728

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6028585, 1.5985498
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2392578, 1.2383518
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2762971, 1.2845044
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3126640, 1.3184853
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4049845, 1.4004045
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.3053656, 1.3016739
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2458372, 1.2498527
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2573247, 1.2544131
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4002581, 1.4188933
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3534317, 1.3530688

Time for backsubstitution: 22.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 220
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 568
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2486
type: DSZ, layer: 3, pos: 2243
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1149
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 715
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 2619
type: DSZ, layer: 3, pos: 1695
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 1144
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 2006
type: DSZ, layer: 3, pos: 1444
type: DSZ, layer: 3, pos: 764
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1496
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 1114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2152

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5735975, upper bound: 0.5687217
time: 5.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5731587, upper bound: 0.5690992
time: 6.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6026173, 1.6006284
1: -13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2378154, 1.2419829
2: -10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.2809150, 1.2724690
3: -14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3174157, 1.3175106
4: 7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4044213, 1.4019594
5: -7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.3018093, 1.2993379
6: -9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2444620, 1.2484298
7: -5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2513304, 1.2499876
8: -4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4065642, 1.4205914
9: -7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3519340, 1.3518815

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2006
type: DSZ, layer: 3, pos: 2619
type: DSZ, layer: 3, pos: 1739
type: DSZ, layer: 3, pos: 1264
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 220
type: DSZ, layer: 3, pos: 1496
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 715
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 408
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1725
type: DSZ, layer: 3, pos: 764
type: DSZ, layer: 3, pos: 568
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 571
type: DSZ, layer: 3, pos: 218
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 1114
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 2243
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 1695
type: DSZ, layer: 3, pos: 718
type: DSZ, layer: 3, pos: 2494
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 1144
type: DSZ, layer: 3, pos: 1397
type: DSZ, layer: 3, pos: 1444
type: DSZ, layer: 3, pos: 2832
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1149

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2006

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5758887, upper bound: 0.5708453
time: 6.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5758804, upper bound: 0.5708538
time: 5.81 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 35.20 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 35.20
Output dim: 4, lower bound: -0.5735975, upper bound: 0.5687217
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 35.20
Output dim: 4, lower bound: -0.5731587, upper bound: 0.5690992
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 35.20
Output dim: 4, lower bound: -0.5758887, upper bound: 0.5708453
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 35.20
Output dim: 4, lower bound: -0.5758804, upper bound: 0.5708538

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 61.05 + 528.22 = 589.27 seconds
