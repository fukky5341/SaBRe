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
execution time: IAR + RelationalAnalysis = 22.59 + 33.86 = 56.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4111793, upper bound: 0.4111788

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 577

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111772, upper bound: 0.4075098
time: 4.82 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075104, upper bound: 0.4111768
time: 4.09 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.93 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.93
Output dim: 7, lower bound: -0.4111772, upper bound: 0.4075098
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.93
Output dim: 7, lower bound: -0.4075104, upper bound: 0.4111768

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2407513, 1.2420000
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3717580, 1.3733284
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3566566, 1.3576534
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9943056, 0.9940506
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2620795, 1.2630366
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0816250, 1.0827165
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2645223, 1.2656338
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9407916, 0.9393435
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0475001, 1.0492530
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2947919, 1.2926888

Time for backsubstitution: 21.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4602

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111755, upper bound: 0.4071260
time: 5.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107929, upper bound: 0.4075082
time: 4.55 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2419996, 1.2407511
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3733287, 1.3717585
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3576536, 1.3566563
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9940505, 0.9943056
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2630365, 1.2620794
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0827169, 1.0816247
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2656333, 1.2645222
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9393435, 0.9407916
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0492527, 1.0475004
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2926886, 1.2947922

Time for backsubstitution: 21.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075086, upper bound: 0.4097909
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4061258, upper bound: 0.4111748
time: 6.09 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.85 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.85
Output dim: 7, lower bound: -0.4111755, upper bound: 0.4071260
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.85
Output dim: 7, lower bound: -0.4107929, upper bound: 0.4075082
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.85
Output dim: 7, lower bound: -0.4075086, upper bound: 0.4097909
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.85
Output dim: 7, lower bound: -0.4061258, upper bound: 0.4111748

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2288971, 1.2349615
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3666301, 1.3702807
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3475122, 1.3522224
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9847927, 0.9883972
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2579179, 1.2560397
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0791233, 1.0812304
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2547019, 1.2598007
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9397867, 0.9376422
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0448289, 1.0447574
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2929485, 1.2895987

Time for backsubstitution: 20.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111735, upper bound: 0.4057405
time: 5.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4097895, upper bound: 0.4071241
time: 5.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2337127, 1.2301459
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3687110, 1.3682003
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3512254, 1.3485093
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9886522, 0.9845380
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2550826, 1.2588754
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0801387, 1.0802152
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2586892, 1.2558136
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9390903, 0.9383386
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0430048, 1.0465815
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2917018, 1.2908454

Time for backsubstitution: 20.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 467

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107914, upper bound: 0.4070088
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4102936, upper bound: 0.4075066
time: 4.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2403831, 1.2363973
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3662772, 1.3691425
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3573623, 1.3558650
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9826679, 0.9900860
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2600746, 1.2540975
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0809231, 1.0809565
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2639275, 1.2599314
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9391916, 0.9403831
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0452476, 1.0460179
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2902830, 1.2882903

Time for backsubstitution: 21.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4585

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068420, upper bound: 0.4097891
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4075068, upper bound: 0.4091239
time: 3.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2376461, 1.2391338
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3707128, 1.3647068
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3568621, 1.3563652
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9898310, 0.9829229
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2550550, 1.2591174
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0820484, 1.0798311
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2610431, 1.2628156
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9389350, 0.9406397
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0477703, 1.0434949
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2861869, 1.2923865

Time for backsubstitution: 20.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4585

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4054591, upper bound: 0.4111730
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4061240, upper bound: 0.4105103
time: 3.44 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.07 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.07
Output dim: 7, lower bound: -0.4111735, upper bound: 0.4057405
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.07
Output dim: 7, lower bound: -0.4097895, upper bound: 0.4071241
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.07
Output dim: 7, lower bound: -0.4107914, upper bound: 0.4070088
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.07
Output dim: 7, lower bound: -0.4102936, upper bound: 0.4075066
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.07
Output dim: 7, lower bound: -0.4068420, upper bound: 0.4097891
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 28.07
Output dim: 7, lower bound: -0.4075068, upper bound: 0.4091239
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.07
Output dim: 7, lower bound: -0.4054591, upper bound: 0.4111730
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.07
Output dim: 7, lower bound: -0.4061240, upper bound: 0.4105103

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2272799, 1.2306075
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3595791, 1.3676660
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3472214, 1.3514311
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9734111, 0.9841762
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2549539, 1.2480563
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0773299, 1.0805619
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2529950, 1.2552099
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9396343, 0.9372337
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0408237, 1.0432749
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2905433, 1.2830966

Time for backsubstitution: 21.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4585

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4105068, upper bound: 0.4057423
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111717, upper bound: 0.4050774
time: 3.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2245433, 1.2333468
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3640151, 1.3632298
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3467207, 1.3519316
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9805741, 0.9770154
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2499342, 1.2530761
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0784552, 1.0794370
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2501111, 1.2580962
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9393783, 0.9374902
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0433469, 1.0407522
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2864466, 1.2871928

Time for backsubstitution: 22.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 467

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4097880, upper bound: 0.4066244
time: 4.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4092903, upper bound: 0.4071221
time: 3.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2321739, 1.2268760
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3599868, 1.3496251
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3213115, 1.3344567
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9873888, 0.9818591
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2356899, 1.2176061
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0741174, 1.0673990
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2529128, 1.2435236
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9370034, 0.9339014
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0408967, 1.0421028
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2884936, 1.2840159

Time for backsubstitution: 21.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6153

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107907, upper bound: 0.4065320
time: 5.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4103288, upper bound: 0.4070081
time: 5.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2304428, 1.2286073
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3501358, 1.3594763
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3371725, 1.3185954
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9859736, 0.9832746
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2138135, 1.2394825
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0673225, 1.0741940
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2463992, 1.2500371
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9346530, 0.9362518
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0385261, 1.0444734
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2848725, 1.2876370

Time for backsubstitution: 21.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 4585
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4102916, upper bound: 0.4061218
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4089078, upper bound: 0.4075047
time: 4.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2416544, 1.2379196
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3477125, 1.3452239
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3322096, 1.3353579
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9662170, 0.9756829
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2350128, 1.2251264
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0629318, 1.0597739
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2616622, 1.2613690
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9405601, 0.9414179
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0440664, 1.0449839
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2324362, 1.2205718

Time for backsubstitution: 21.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4066304, upper bound: 0.4094947
time: 3.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4065466, upper bound: 0.4095791
time: 4.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2389178, 1.2406561
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3521481, 1.3407881
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3317094, 1.3358581
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9733803, 0.9685197
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2299931, 1.2301462
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0640571, 1.0586486
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2587783, 1.2642531
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9403036, 0.9416745
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0465891, 1.0424612
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2283397, 1.2246680

Time for backsubstitution: 21.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4052476, upper bound: 0.4108821
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4051636, upper bound: 0.4109663
time: 3.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2391686, 1.2404053
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3467937, 1.3461423
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3363552, 1.3312125
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9754279, 0.9664721
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2260835, 1.2340558
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0608659, 1.0618398
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2624805, 1.2605506
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9399698, 0.9420083
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0467362, 1.0423141
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2184684, 1.2345393

Time for backsubstitution: 21.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 4602
type: DSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4059126, upper bound: 0.4102159
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058286, upper bound: 0.4103014
time: 3.44 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.07 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4105068, upper bound: 0.4057423
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4111717, upper bound: 0.4050774
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4097880, upper bound: 0.4066244
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4092903, upper bound: 0.4071221
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4107907, upper bound: 0.4065320
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4103288, upper bound: 0.4070081
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4102916, upper bound: 0.4061218
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4089078, upper bound: 0.4075047
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4066304, upper bound: 0.4094947
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4065466, upper bound: 0.4095791
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4052476, upper bound: 0.4108821
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4051636, upper bound: 0.4109663
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4059126, upper bound: 0.4102159
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.07
Output dim: 7, lower bound: -0.4058286, upper bound: 0.4103014

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2285514, 1.2321301
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3410144, 1.3437471
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3220682, 1.3309238
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9569607, 0.9697732
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2298920, 1.2190849
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0593383, 1.0593791
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2507308, 1.2566479
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9410033, 0.9382688
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0396426, 1.0422409
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2326965, 1.2153783

Time for backsubstitution: 21.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 467
type: DSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4102968, upper bound: 0.4054435
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4102126, upper bound: 0.4055272
time: 3.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2288024, 1.2318792
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3356605, 1.3491013
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3267140, 1.3262782
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9590082, 0.9677258
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2259824, 1.2229943
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0561471, 1.0625703
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2544334, 1.2529454
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9406695, 0.9386026
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0397897, 1.0420938
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2228251, 1.2252498

Time for backsubstitution: 21.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111710, upper bound: 0.4046293
time: 3.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107083, upper bound: 0.4050757
time: 3.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2230048, 1.2300774
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3552904, 1.3446546
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3168073, 1.3378787
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9793108, 0.9743366
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2305419, 1.2118068
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0724339, 1.0666208
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2443347, 1.2458062
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9372914, 0.9330530
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0412388, 1.0362735
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2832384, 1.2803636

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4095755, upper bound: 0.4066228
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4092923, upper bound: 0.4066273
time: 3.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2212737, 1.2318087
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3454399, 1.3545058
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3326683, 1.3220177
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9778955, 0.9757521
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2086651, 1.2336833
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0656390, 1.0734158
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2378211, 1.2523197
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9349411, 0.9354033
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0388682, 1.0386441
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2796173, 1.2839847

Time for backsubstitution: 21.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6153
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4092896, upper bound: 0.4066524
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4088464, upper bound: 0.4071214
time: 4.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2315087, 1.2281971
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3574336, 1.3464999
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3121738, 1.3288081
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9883316, 0.9826331
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2119360, 1.1992643
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0469985, 1.0364046
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2402694, 1.2367098
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9347003, 0.9302918
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0461216, 1.0439823
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2697661, 1.2620585

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107886, upper bound: 0.4051792
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4094048, upper bound: 0.4065330
time: 3.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.4819927, -9.2464151, -11.4819927, -9.2464151, -1.2334950, 1.2262110
1: -6.5261440, -4.7152486, -6.5261440, -4.7152486, -1.3568614, 1.3470714
2: -6.2376761, -4.2180405, -6.2376761, -4.2180405, -1.3156624, 1.3253191
3: -5.3569956, -3.7469785, -5.3569956, -3.7469785, -0.9881630, 0.9828018
4: -7.4061117, -5.1482801, -7.4061117, -5.1482801, -1.2173476, 1.1938524
5: -10.4922600, -8.6001883, -10.4922600, -8.6001883, -1.0431228, 1.0402801
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2460992, 1.2308805
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9333937, 0.9315982
8: -6.4546919, -4.6735840, -6.4546919, -4.6735840, -1.0427761, 1.0473280
9: -5.4519520, -3.7852185, -5.4519520, -3.7852185, -1.2665365, 1.2652876

Time for backsubstitution: 22.36 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.44 + 563.12 = 619.57 seconds
