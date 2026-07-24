## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00056538


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005468, 0.0005468)
1: (0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010591, 0.0010591)
2: (-0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0085423, 0.0085423)
3: (-0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007630, 0.0007630)
4: (0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0037017, 0.0037017)
5: (-0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005526, 0.0005526)
6: (0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0010135, 0.0010135)
7: (0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0067008, 0.0067008)
8: (0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020993, 0.0020993)
9: (-0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041899, 0.0041899)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.00 + 2.32 = 3.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0006282, upper bound: 0.0006282

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006186, upper bound: 0.0006056
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006056, upper bound: 0.0006186
time: 1.32 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.67 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.67
Output dim: 6, lower bound: -0.0006186, upper bound: 0.0006056
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.67
Output dim: 6, lower bound: -0.0006056, upper bound: 0.0006186

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005464, 0.0005462
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010584, 0.0010580
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0085339, 0.0085371
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007625, 0.0007622
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036994, 0.0036981
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005520, 0.0005522
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0010129, 0.0010125
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0066966, 0.0066942
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020980, 0.0020972
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041858, 0.0041873

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006095, upper bound: 0.0005961
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006094, upper bound: 0.0005967
time: 1.45 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005462, 0.0005464
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010580, 0.0010584
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0085371, 0.0085339
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007622, 0.0007625
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036981, 0.0036994
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005522, 0.0005520
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0010125, 0.0010129
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0066942, 0.0066966
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020972, 0.0020980
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041873, 0.0041858

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005962, upper bound: 0.0006094
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005962, upper bound: 0.0006095
time: 1.34 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.34 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.34
Output dim: 6, lower bound: -0.0006095, upper bound: 0.0005961
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.34
Output dim: 6, lower bound: -0.0006094, upper bound: 0.0005967
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.34
Output dim: 6, lower bound: -0.0005962, upper bound: 0.0006094
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.34
Output dim: 6, lower bound: -0.0005962, upper bound: 0.0006095

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005439, 0.0005438
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010534, 0.0010533
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0084962, 0.0084967
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007589, 0.0007588
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036819, 0.0036817
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005496, 0.0005496
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0010081, 0.0010080
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0066649, 0.0066646
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020881, 0.0020880
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041673, 0.0041675

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006021, upper bound: 0.0005777
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005877, upper bound: 0.0005886
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005440, 0.0005437
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010537, 0.0010530
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0084936, 0.0084993
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007591, 0.0007586
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036831, 0.0036806
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005494, 0.0005498
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0010084, 0.0010077
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0066670, 0.0066625
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020887, 0.0020873
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041660, 0.0041688

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005318, upper bound: 0.0005230
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005318, upper bound: 0.0005230
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005437, 0.0005440
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010530, 0.0010537
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0084993, 0.0084936
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007586, 0.0007591
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036806, 0.0036831
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005498, 0.0005494
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0010077, 0.0010084
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0066625, 0.0066670
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020873, 0.0020887
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041688, 0.0041660

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005230, upper bound: 0.0005318
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005230, upper bound: 0.0005318
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005438, 0.0005439
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010533, 0.0010534
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0084967, 0.0084962
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007588, 0.0007589
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036817, 0.0036819
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005496, 0.0005496
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0010080, 0.0010081
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0066646, 0.0066649
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020880, 0.0020881
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041675, 0.0041673

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005796, upper bound: 0.0005746
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005658, upper bound: 0.0005948
time: 1.46 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.65 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0006021, upper bound: 0.0005777
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0005877, upper bound: 0.0005886
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0005318, upper bound: 0.0005230
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0005318, upper bound: 0.0005230
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0005230, upper bound: 0.0005318
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0005230, upper bound: 0.0005318
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0005796, upper bound: 0.0005746
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 6, lower bound: -0.0005658, upper bound: 0.0005948

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005386, 0.0005372
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010433, 0.0010406
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0083934, 0.0084151
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007516, 0.0007497
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036466, 0.0036372
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005430, 0.0005444
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009984, 0.0009958
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0066010, 0.0065839
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020680, 0.0020627
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041169, 0.0041275

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005951, upper bound: 0.0005619
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005619, upper bound: 0.0005706
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005373, 0.0005386
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010407, 0.0010432
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0084144, 0.0083938
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007497, 0.0007515
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036374, 0.0036463
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005443, 0.0005430
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009959, 0.0009983
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0065843, 0.0066004
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020628, 0.0020679
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041272, 0.0041171

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005808, upper bound: 0.0005724
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005619, upper bound: 0.0005815
time: 1.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005223, 0.0005156
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010116, 0.0009988
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0080559, 0.0081596
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007288, 0.0007195
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0035359, 0.0034909
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005211, 0.0005278
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009681, 0.0009558
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0064005, 0.0063192
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020052, 0.0019798
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0039513, 0.0040022

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005689, upper bound: 0.0005478
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005497, upper bound: 0.0005631
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005156, 0.0005223
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009987, 0.0010116
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0081593, 0.0080554
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007195, 0.0007287
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0034907, 0.0035357
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005278, 0.0005211
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009557, 0.0009680
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0063188, 0.0064003
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0019796, 0.0020052
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0040020, 0.0039511

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005514, upper bound: 0.0005584
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005376, upper bound: 0.0005825
time: 1.36 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.52 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 6, lower bound: -0.0005951, upper bound: 0.0005619
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 6, lower bound: -0.0005619, upper bound: 0.0005706
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 6, lower bound: -0.0005808, upper bound: 0.0005724
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 6, lower bound: -0.0005619, upper bound: 0.0005815
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 6, lower bound: -0.0005689, upper bound: 0.0005478
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.52
Output dim: 6, lower bound: -0.0005497, upper bound: 0.0005631
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.52
Output dim: 6, lower bound: -0.0005514, upper bound: 0.0005584
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 6, lower bound: -0.0005376, upper bound: 0.0005825

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005369, 0.0005330
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010400, 0.0010323
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0083268, 0.0083882
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007492, 0.0007437
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036349, 0.0036083
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005386, 0.0005426
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009952, 0.0009879
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0065798, 0.0065317
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020614, 0.0020463
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0040842, 0.0041143

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005843, upper bound: 0.0005331
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005641, upper bound: 0.0005514
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005344, 0.0005356
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010350, 0.0010375
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0083682, 0.0083485
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007456, 0.0007474
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036177, 0.0036263
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005413, 0.0005401
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009905, 0.0009928
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0065487, 0.0065641
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020517, 0.0020565
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041045, 0.0040949

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005489, upper bound: 0.0005578
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005494, upper bound: 0.0005576
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005356, 0.0005343
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010374, 0.0010350
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0083478, 0.0083675
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007473, 0.0007456
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036259, 0.0036174
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005400, 0.0005413
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009927, 0.0009904
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0065636, 0.0065482
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020563, 0.0020515
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0040945, 0.0041042

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005807, upper bound: 0.0005703
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005781, upper bound: 0.0005723
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005330, 0.0005370
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010324, 0.0010401
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0083891, 0.0083272
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007437, 0.0007493
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036085, 0.0036353
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005427, 0.0005387
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009880, 0.0009953
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0065320, 0.0065806
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020464, 0.0020616
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041148, 0.0040844

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005489, upper bound: 0.0005685
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005489, upper bound: 0.0005685
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005204, 0.0005124
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010079, 0.0009926
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0080058, 0.0081300
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007261, 0.0007150
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0035230, 0.0034692
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005179, 0.0005259
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009646, 0.0009498
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0063773, 0.0062799
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0019980, 0.0019675
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0039268, 0.0039877

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005622, upper bound: 0.0005252
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005215, upper bound: 0.0005409
time: 1.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0004892, 0.0005039
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0009476, 0.0009760
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0078725, 0.0076434
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0006827, 0.0007031
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0033122, 0.0034115
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005093, 0.0004944
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009068, 0.0009340
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0059956, 0.0061753
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0018784, 0.0019347
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0038614, 0.0037490

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004691, upper bound: 0.0004994
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004691, upper bound: 0.0004994
time: 1.19 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 6, lower bound: -0.0005843, upper bound: 0.0005331
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 6, lower bound: -0.0005641, upper bound: 0.0005514
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 6, lower bound: -0.0005489, upper bound: 0.0005578
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 6, lower bound: -0.0005494, upper bound: 0.0005576
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 6, lower bound: -0.0005807, upper bound: 0.0005703
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 6, lower bound: -0.0005781, upper bound: 0.0005723
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 6, lower bound: -0.0005489, upper bound: 0.0005685
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 6, lower bound: -0.0005489, upper bound: 0.0005685
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 6, lower bound: -0.0005622, upper bound: 0.0005252
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 6, lower bound: -0.0005215, upper bound: 0.0005409
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 6, lower bound: -0.0004691, upper bound: 0.0004994
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.22
Output dim: 6, lower bound: -0.0004691, upper bound: 0.0004994

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005349, 0.0005296
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010360, 0.0010259
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0082746, 0.0083566
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007464, 0.0007390
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036213, 0.0035857
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005353, 0.0005406
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009915, 0.0009817
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0065551, 0.0064908
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020537, 0.0020335
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0040586, 0.0040988

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005099, upper bound: 0.0005000
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005099, upper bound: 0.0005208
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005390, 0.0005368
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010441, 0.0010396
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0083856, 0.0084213
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007521, 0.0007490
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036493, 0.0036338
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005425, 0.0005448
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009991, 0.0009949
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0066058, 0.0065778
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020696, 0.0020608
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041131, 0.0041306

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005652, upper bound: 0.0005371
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005467, upper bound: 0.0005552
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005380, 0.0005378
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010421, 0.0010416
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0084016, 0.0084053
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007507, 0.0007504
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0036423, 0.0036408
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005435, 0.0005437
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009972, 0.0009968
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0065932, 0.0065904
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020656, 0.0020647
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041209, 0.0041227

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005649, upper bound: 0.0005591
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005649, upper bound: 0.0005591
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005250, 0.0005280
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010169, 0.0010228
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0082496, 0.0082026
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007326, 0.0007368
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0035545, 0.0035749
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005337, 0.0005306
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009732, 0.0009788
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0064342, 0.0064711
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020158, 0.0020274
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0040463, 0.0040233

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005153, upper bound: 0.0005385
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005153, upper bound: 0.0005519
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005241, 0.0005290
1: 0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010152, 0.0010246
2: -0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0082644, 0.0081886
3: -0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007314, 0.0007381
4: 0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0035485, 0.0035813
5: -0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005346, 0.0005297
6: 0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0009715, 0.0009805
7: 0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0064233, 0.0064828
8: 0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020124, 0.0020310
9: -0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0040536, 0.0040164

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005050, upper bound: 0.0005220
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005050, upper bound: 0.0005498
time: 1.37 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.83 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 6, lower bound: -0.0005099, upper bound: 0.0005000
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 6, lower bound: -0.0005099, upper bound: 0.0005208
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 6, lower bound: -0.0005652, upper bound: 0.0005371
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 6, lower bound: -0.0005467, upper bound: 0.0005552
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 6, lower bound: -0.0005649, upper bound: 0.0005591
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 6, lower bound: -0.0005649, upper bound: 0.0005591
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 6, lower bound: -0.0005153, upper bound: 0.0005385
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 6, lower bound: -0.0005153, upper bound: 0.0005519
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 6, lower bound: -0.0005050, upper bound: 0.0005220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 6, lower bound: -0.0005050, upper bound: 0.0005498

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.32 + 82.61 = 85.94 seconds
