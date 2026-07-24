## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.487812356


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1203337, 3.1203332)
1: (-9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8023891, 2.8023882)
2: (-4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8518019, 2.8518019)
3: (-1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933)
4: (-14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2965384, 3.2965384)
5: (-8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1671762, 2.1671762)
6: (-12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2981486, 3.2981482)
7: (-9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8508062, 2.8508062)
8: (9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6844096, 2.6844096)
9: (-7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9997826, 2.9997826)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.73 + 37.14 = 59.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.4952888, upper bound: 1.4952880

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6253

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952866, upper bound: 1.4934429
time: 7.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4934440, upper bound: 1.4952867
time: 7.52 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 14.78 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 14.78
Output dim: 8, lower bound: -1.4952866, upper bound: 1.4934429
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 14.78
Output dim: 8, lower bound: -1.4934440, upper bound: 1.4952867

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1221218, 3.1218376
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8045897, 2.8050079
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8518000, 2.8520122
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3139572, 3.3202295
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1638970, 2.1643069
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2882824, 3.2895150
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8541689, 2.8573694
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6780953, 2.6751938
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9976783, 2.9973783

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 931

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4931811, upper bound: 1.4916529
time: 8.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4934954, upper bound: 1.4913383
time: 8.56 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1218376, 3.1221218
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8050075, 2.8045902
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8520117, 2.8518009
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3202295, 3.3139577
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1643066, 2.1638970
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2895145, 3.2882819
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8573694, 2.8541694
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6751933, 2.6780958
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9973779, 2.9976783

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5762

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4900102, upper bound: 1.4952843
time: 7.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4934415, upper bound: 1.4918527
time: 6.19 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 35.63 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 35.63
Output dim: 8, lower bound: -1.4931811, upper bound: 1.4916529
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 35.63
Output dim: 8, lower bound: -1.4934954, upper bound: 1.4913383
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 35.63
Output dim: 8, lower bound: -1.4900102, upper bound: 1.4952843
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 35.63
Output dim: 8, lower bound: -1.4934415, upper bound: 1.4918527

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1220007, 3.1217966
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8036761, 2.8047090
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8515739, 2.8513198
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3127651, 3.3198342
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1638246, 2.1640875
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2875118, 3.2892685
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8528414, 2.8569293
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6778479, 2.6744442
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9969277, 2.9971304

Time for backsubstitution: 22.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 832

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4926831, upper bound: 1.4914364
time: 5.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4929705, upper bound: 1.4911493
time: 4.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1220808, 3.1217165
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8042912, 2.8040929
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8511086, 2.8517847
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3135614, 3.3190370
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1636777, 2.1642342
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2880344, 3.2887459
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8537292, 2.8560419
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6773462, 2.6749461
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9974303, 2.9966269

Time for backsubstitution: 21.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5845

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4917701, upper bound: 1.4899239
time: 6.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4934889, upper bound: 1.4899243
time: 15.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.0856991, 3.0808129
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8026352, 2.8018780
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8468318, 2.8472657
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3837504, 3.3842940
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3107681, 3.3031425
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1443176, 2.1410654
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2955284, 3.2917128
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8628597, 2.8636961
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6654997, 2.6696115
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9999771, 3.0008540

Time for backsubstitution: 21.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 832

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4895058, upper bound: 1.4950730
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4897930, upper bound: 1.4947854
time: 4.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.0805292, 3.0859833
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8022957, 2.8022175
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8474765, 2.8466206
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3860192, 3.3820252
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3094149, 3.3044963
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1414752, 2.1439078
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2929459, 3.2942948
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8668957, 2.8596601
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6667089, 2.6684020
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -3.0005531, 3.0002766

Time for backsubstitution: 23.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 931

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4627

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918227, upper bound: 1.4918469
time: 10.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4934355, upper bound: 1.4902341
time: 8.28 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 42.08 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 42.08
Output dim: 8, lower bound: -1.4926831, upper bound: 1.4914364
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 42.08
Output dim: 8, lower bound: -1.4929705, upper bound: 1.4911493
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 42.08
Output dim: 8, lower bound: -1.4917701, upper bound: 1.4899239
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 42.08
Output dim: 8, lower bound: -1.4934889, upper bound: 1.4899243
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 42.08
Output dim: 8, lower bound: -1.4895058, upper bound: 1.4950730
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 42.08
Output dim: 8, lower bound: -1.4897930, upper bound: 1.4947854
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 42.08
Output dim: 8, lower bound: -1.4918227, upper bound: 1.4918469
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 42.08
Output dim: 8, lower bound: -1.4934355, upper bound: 1.4902341

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1218128, 3.1216607
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8036280, 2.8046451
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8516512, 2.8513837
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3135233, 3.3207693
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1629376, 2.1628733
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2864418, 3.2878017
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8527460, 2.8568592
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6759048, 2.6727426
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9959631, 2.9964256

Time for backsubstitution: 23.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4630

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4926827, upper bound: 1.4913366
time: 5.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4925835, upper bound: 1.4914357
time: 5.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1218643, 3.1216102
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8036118, 2.8046603
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8516378, 2.8513975
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3136997, 3.3205929
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1626101, 2.1632009
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2860460, 3.2881970
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8527708, 2.8568335
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6761465, 2.6725011
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9962225, 2.9961653

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 848

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4891742, upper bound: 1.4905508
time: 4.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4923729, upper bound: 1.4873523
time: 4.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1176739, 3.1156096
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7996707, 2.8008819
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8402767, 2.8362026
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2987366, 3.3087330
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1597161, 2.1611876
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2833962, 3.2835765
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8415108, 2.8475490
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6660662, 2.6660881
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9954858, 2.9951382

Time for backsubstitution: 23.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 832

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4912724, upper bound: 1.4897077
time: 5.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4915598, upper bound: 1.4894205
time: 4.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1159744, 3.1174717
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8010802, 2.7994728
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8355274, 2.8409534
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3032608, 3.3042111
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1609240, 2.1602726
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2828660, 3.2851496
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8452396, 2.8438230
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6694946, 2.6636662
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9959416, 2.9955940

Time for backsubstitution: 23.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 5762
type: DSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4627

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918708, upper bound: 1.4899183
time: 8.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4934827, upper bound: 1.4883054
time: 5.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.0855112, 3.0806761
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8025866, 2.8018141
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8469105, 2.8473306
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3834486, 3.3840742
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3115263, 3.3040767
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1434302, 2.1398506
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2944579, 3.2902470
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8627653, 2.8636279
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6635561, 2.6679091
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9990106, 3.0001478

Time for backsubstitution: 23.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4627

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4878936, upper bound: 1.4950671
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4894998, upper bound: 1.4934612
time: 6.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.0855627, 3.0806255
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8025713, 2.8018298
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8468962, 2.8473439
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3835306, 3.3839927
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3117027, 3.3039007
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1431031, 2.1401782
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2940631, 3.2906427
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8627920, 2.8636022
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6637979, 2.6676674
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9992709, 2.9998875

Time for backsubstitution: 23.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 848

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4881976, upper bound: 1.4941885
time: 4.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4878486, upper bound: 1.4897741
time: 4.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.0738144, 3.0767632
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7979169, 2.7962203
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8390365, 2.8350449
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3799348, 3.3775878
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3037529, 3.2967482
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1396356, 2.1425631
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2921252, 3.2931705
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8597050, 2.8498130
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6584620, 2.6623795
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -3.0004239, 3.0001187

Time for backsubstitution: 25.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4630

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918222, upper bound: 1.4917484
time: 8.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4917236, upper bound: 1.4918465
time: 7.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.0713100, 3.0792685
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7962985, 2.7978392
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8359017, 2.8381796
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3815808, 3.3759408
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3016663, 3.2988338
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1401305, 2.1420684
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2918220, 3.2934742
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8570480, 2.8524694
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6606870, 2.6601543
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -3.0003953, 3.0001459

Time for backsubstitution: 25.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5746

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930441, upper bound: 1.4902339
time: 5.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4934351, upper bound: 1.4898424
time: 5.14 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 35.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4926827, upper bound: 1.4913366
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4925835, upper bound: 1.4914357
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4891742, upper bound: 1.4905508
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4923729, upper bound: 1.4873523
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4912724, upper bound: 1.4897077
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4915598, upper bound: 1.4894205
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4918708, upper bound: 1.4899183
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4934827, upper bound: 1.4883054
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4878936, upper bound: 1.4950671
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4894998, upper bound: 1.4934612
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4881976, upper bound: 1.4941885
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4878486, upper bound: 1.4897741
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4918222, upper bound: 1.4917484
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4917236, upper bound: 1.4918465
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4930441, upper bound: 1.4902339
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 35.50
Output dim: 8, lower bound: -1.4934351, upper bound: 1.4898424

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1149073, 3.1156192
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7982383, 2.7999282
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8483090, 2.8492837
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.3165388, 3.3233953
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1580563, 2.1572347
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2799401, 3.2821140
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8502522, 2.8540096
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6744118, 2.6710372
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9916201, 2.9926238

Time for backsubstitution: 22.90 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.87 + 544.53 = 604.40 seconds
