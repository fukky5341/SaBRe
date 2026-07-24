## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.4091261895


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2431149, 1.2431149)
1: (-6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3747201, 1.3747203)
2: (-6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3585410, 1.3585410)
3: (-5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9945278, 0.9945278)
4: (-7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2638829, 1.2638830)
5: (-10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0836921, 1.0836918)
6: (-17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2666290, 1.2666286)
7: (5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9420798, 0.9420798)
8: (-6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0508149, 1.0508149)
9: (-5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2966568, 1.2966571)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.02 + 33.38 = 56.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4111793, upper bound: 0.4111788

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6153

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111786, upper bound: 0.4107158
time: 4.76 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107165, upper bound: 0.4111782
time: 4.84 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.70 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.70
Output dim: 7, lower bound: -0.4111786, upper bound: 0.4107158
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.70
Output dim: 7, lower bound: -0.4107165, upper bound: 0.4111782

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2424500, 1.2444361
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3721666, 1.3715949
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3494048, 1.3528934
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9954705, 0.9953020
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2401295, 1.2455409
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0565729, 1.0526974
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2539854, 1.2598145
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9397764, 0.9384699
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0560396, 1.0526941
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2779286, 1.2746994

Time for backsubstitution: 21.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4585

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4105120, upper bound: 0.4107139
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111768, upper bound: 0.4100501
time: 3.49 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2444360, 1.2424500
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3715949, 1.3721664
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3528934, 1.3494048
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9953020, 0.9954706
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2455406, 1.2401291
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0526974, 1.0565727
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2598147, 1.2539852
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9384699, 0.9397763
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0526941, 1.0560396
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2746994, 1.2779286

Time for backsubstitution: 21.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4585

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100505, upper bound: 0.4111760
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107145, upper bound: 0.4105147
time: 3.45 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.67 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.67
Output dim: 7, lower bound: -0.4105120, upper bound: 0.4107139
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.67
Output dim: 7, lower bound: -0.4111768, upper bound: 0.4100501
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.67
Output dim: 7, lower bound: -0.4100505, upper bound: 0.4111760
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.67
Output dim: 7, lower bound: -0.4107145, upper bound: 0.4105147

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2437203, 1.2459570
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3535793, 1.3476589
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3242121, 1.3323359
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9790211, 0.9808997
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2151055, 1.2166078
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0385625, 1.0315039
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2516785, 1.2612023
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9411457, 0.9395056
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0548592, 1.0516608
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2200272, 1.2069454

Time for backsubstitution: 21.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 577

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4105099, upper bound: 0.4070350
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068431, upper bound: 0.4107123
time: 3.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2439711, 1.2457064
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3482301, 1.3530128
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3288574, 1.3277006
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9810686, 0.9788522
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2111959, 1.2205189
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0353794, 1.0346954
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2553811, 1.2575077
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9408121, 0.9398394
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0550067, 1.0515137
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2101746, 1.2168167

Time for backsubstitution: 21.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 577

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111747, upper bound: 0.4063711
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075079, upper bound: 0.4100479
time: 3.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2457063, 1.2439711
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3530128, 1.3482304
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3277006, 1.3288574
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9788523, 0.9810685
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2205191, 1.2111961
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0346954, 1.0353794
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2575078, 1.2553809
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9398394, 0.9408121
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0515137, 1.0550067
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2168169, 1.2101748

Time for backsubstitution: 21.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 577

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100484, upper bound: 0.4075107
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4063717, upper bound: 0.4111741
time: 4.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2459569, 1.2437203
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3476589, 1.3535793
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3323359, 1.3242121
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9808998, 0.9790208
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2166080, 1.2151055
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0315039, 1.0385625
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2612023, 1.2516783
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9395056, 0.9411458
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0516608, 1.0548594
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2069454, 1.2200272

Time for backsubstitution: 21.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 577

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107125, upper bound: 0.4068425
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4070357, upper bound: 0.4105094
time: 3.87 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.00 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.4105099, upper bound: 0.4070350
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.4068431, upper bound: 0.4107123
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.4111747, upper bound: 0.4063711
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.4075079, upper bound: 0.4100479
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.4100484, upper bound: 0.4075107
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.4063717, upper bound: 0.4111741
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.4107125, upper bound: 0.4068425
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.4070357, upper bound: 0.4105094

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2413566, 1.2448422
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3506172, 1.3462670
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3223271, 1.3314478
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9787982, 0.9804220
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2133017, 1.2157615
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0364954, 1.0305285
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2495723, 1.2602077
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9398577, 0.9367695
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0515447, 1.0500989
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2181623, 1.2029772

Time for backsubstitution: 21.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4105081, upper bound: 0.4056815
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4091241, upper bound: 0.4070333
time: 3.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2426054, 1.2435933
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3521874, 1.3446970
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3233242, 1.3304508
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9785433, 0.9806771
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2142591, 1.2148042
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0375874, 1.0294368
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2506838, 1.2590960
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9384096, 0.9382176
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0532973, 1.0483463
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2160590, 1.2050805

Time for backsubstitution: 21.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068413, upper bound: 0.4093452
time: 4.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4054585, upper bound: 0.4107131
time: 3.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2416074, 1.2445917
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3452685, 1.3516209
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3269725, 1.3268127
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9808459, 0.9783745
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2093925, 1.2196727
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0333123, 1.0337200
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2532744, 1.2565131
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9395242, 0.9371033
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0516922, 1.0499516
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2083097, 1.2128484

Time for backsubstitution: 21.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111729, upper bound: 0.4050155
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4097889, upper bound: 0.4063722
time: 3.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2428563, 1.2433428
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3468382, 1.3500512
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3279696, 1.3258154
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9805908, 0.9786296
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2103496, 1.2187153
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0344043, 1.0326281
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2543864, 1.2554014
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9380760, 0.9385514
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0534446, 1.0481992
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2062066, 1.2149518

Time for backsubstitution: 21.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4075061, upper bound: 0.4086807
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4061233, upper bound: 0.4100490
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2433426, 1.2428564
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3500512, 1.3468385
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3258157, 1.3279696
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9786296, 0.9805909
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2187152, 1.2103497
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0326281, 1.0344043
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2554016, 1.2543862
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9385514, 0.9380760
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0481989, 1.0534449
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2149518, 1.2062066

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100463, upper bound: 0.4061260
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4086811, upper bound: 0.4075054
time: 3.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2445917, 1.2416074
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3516209, 1.3452685
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3268127, 1.3269725
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9783745, 0.9808459
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2196727, 1.2093924
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0337200, 1.0333123
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2565131, 1.2532746
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9371033, 0.9395242
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0499516, 1.0516922
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2128484, 1.2083097

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4063695, upper bound: 0.4097883
time: 4.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050143, upper bound: 0.4111724
time: 3.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2435935, 1.2426054
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3446968, 1.3521874
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3304510, 1.3233240
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9806771, 0.9785433
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2148042, 1.2142591
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0294371, 1.0375872
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2590961, 1.2506837
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9382176, 0.9384097
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0483463, 1.0532973
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2050805, 1.2160590

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107103, upper bound: 0.4054612
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4093456, upper bound: 0.4068406
time: 3.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2448423, 1.2413566
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3462670, 1.3506174
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3314481, 1.3223269
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9804220, 0.9787984
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2157617, 1.2133019
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0305288, 1.0364954
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2602077, 1.2495720
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9367695, 0.9398577
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0500987, 1.0515447
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2029772, 1.2181623

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4070336, upper bound: 0.4091235
time: 4.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4056788, upper bound: 0.4105074
time: 3.95 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.56 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4105081, upper bound: 0.4056815
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4091241, upper bound: 0.4070333
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4068413, upper bound: 0.4093452
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4054585, upper bound: 0.4107131
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4111729, upper bound: 0.4050155
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4097889, upper bound: 0.4063722
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4075061, upper bound: 0.4086807
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4061233, upper bound: 0.4100490
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4100463, upper bound: 0.4061260
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4086811, upper bound: 0.4075054
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4063695, upper bound: 0.4097883
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4050143, upper bound: 0.4111724
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4107103, upper bound: 0.4054612
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4093456, upper bound: 0.4068406
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4070336, upper bound: 0.4091235
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.56
Output dim: 7, lower bound: -0.4056788, upper bound: 0.4105074

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2397394, 1.2404884
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3435647, 1.3436503
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3220353, 1.3306561
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9674149, 0.9762018
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2103405, 1.2077801
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0347016, 1.0298600
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2478652, 1.2556169
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9397058, 0.9363611
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0475395, 1.0486162
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2157564, 1.1964753

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 467

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4105066, upper bound: 0.4051757
time: 4.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100088, upper bound: 0.4056769
time: 3.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2409883, 1.2392396
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3451355, 1.3420804
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3230324, 1.3296590
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9671597, 0.9764569
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2112980, 1.2068229
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0357935, 1.0287683
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2489772, 1.2545053
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9382577, 0.9378091
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0492918, 1.0468638
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2136531, 1.1985784

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 467

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4068398, upper bound: 0.4088490
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4063420, upper bound: 0.4093469
time: 3.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2382517, 1.2419761
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3495710, 1.3376446
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3225322, 1.3301592
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9743228, 0.9692937
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2062778, 1.2118427
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0369186, 1.0276430
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2460928, 1.2573895
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9380012, 0.9380658
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0518148, 1.0443408
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2095571, 1.2026746

Time for backsubstitution: 22.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 467

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4054570, upper bound: 0.4102089
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4049593, upper bound: 0.4107083
time: 4.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2399902, 1.2402380
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3382165, 1.3490045
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3266811, 1.3260207
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9694624, 0.9741542
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2064309, 1.2116913
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0315185, 1.0330515
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2515678, 1.2519224
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9393721, 0.9366949
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0476868, 1.0484691
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2059038, 1.2063465

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 467

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111714, upper bound: 0.4045118
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106736, upper bound: 0.4050128
time: 3.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2372537, 1.2429745
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3426521, 1.3445687
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3261809, 1.3265209
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9766254, 0.9669912
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2014112, 1.2167112
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0326438, 1.0319262
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2486839, 1.2548065
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9391155, 0.9369514
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0502098, 1.0459464
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2018075, 1.2104428

Time for backsubstitution: 21.51 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.40 + 557.28 = 613.69 seconds
