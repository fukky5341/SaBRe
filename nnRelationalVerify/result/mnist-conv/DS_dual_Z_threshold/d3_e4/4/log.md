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
execution time: IAR + RelationalAnalysis = 23.24 + 37.12 = 60.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.4952888, upper bound: 1.4952880

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4627

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4936705, upper bound: 1.4952827
time: 7.86 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952828, upper bound: 1.4936694
time: 5.36 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 13.32 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 13.32
Output dim: 8, lower bound: -1.4936705, upper bound: 1.4952827
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 13.32
Output dim: 8, lower bound: -1.4952828, upper bound: 1.4936694

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1136179, 3.1111126
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7980108, 2.7963924
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8433599, 2.8402262
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2908764, 3.2887897
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1653371, 2.1658318
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2973285, 3.2970238
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8436165, 2.8409595
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6761622, 2.6783879
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9996519, 2.9996257

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4630

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4936700, upper bound: 1.4951843
time: 6.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4935715, upper bound: 1.4952823
time: 7.33 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1111135, 3.1136179
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7963915, 2.7980113
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8402262, 2.8433609
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2887897, 3.2908759
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1658316, 2.1653371
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2970233, 3.2973275
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8409595, 2.8436160
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6783876, 2.6761627
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9996252, 2.9996524

Time for backsubstitution: 21.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4630
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4630

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952823, upper bound: 1.4935704
time: 5.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4951843, upper bound: 1.4936699
time: 10.12 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 37.50 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 37.50
Output dim: 8, lower bound: -1.4936700, upper bound: 1.4951843
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 37.50
Output dim: 8, lower bound: -1.4935715, upper bound: 1.4952823
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 37.50
Output dim: 8, lower bound: -1.4952823, upper bound: 1.4935704
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 37.50
Output dim: 8, lower bound: -1.4951843, upper bound: 1.4936699

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1067152, 3.1050749
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7926207, 2.7916746
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8400187, 2.8381262
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2938871, 3.2914128
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1604552, 2.1601925
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2908268, 3.2913351
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8411236, 2.8381104
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6746702, 2.6766825
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9953098, 2.9958229

Time for backsubstitution: 21.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5746

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4932782, upper bound: 1.4951834
time: 5.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4936696, upper bound: 1.4947928
time: 5.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1075802, 3.1042099
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7932940, 2.7910018
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8412604, 2.8368845
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2934990, 3.2918015
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1596980, 2.1609497
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2916393, 3.2905221
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8407669, 2.8384666
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6744571, 2.6768951
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9958496, 2.9952822

Time for backsubstitution: 21.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5746

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4931797, upper bound: 1.4952815
time: 4.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4935711, upper bound: 1.4948901
time: 5.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1042099, 3.1075797
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7910013, 2.7932935
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8368840, 2.8412604
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2918015, 3.2934990
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1609497, 2.1596980
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2905216, 3.2916389
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8384666, 2.8407669
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6768951, 2.6744573
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9952822, 2.9958501

Time for backsubstitution: 21.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5746

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4948905, upper bound: 1.4935701
time: 5.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952819, upper bound: 1.4931788
time: 5.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1050749, 3.1067152
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7916746, 2.7926211
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8381257, 2.8400187
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2914133, 3.2938871
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1601925, 2.1604550
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2913361, 3.2908258
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8381100, 2.8411231
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6766825, 2.6746702
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9958229, 2.9953094

Time for backsubstitution: 21.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5746

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4947925, upper bound: 1.4936689
time: 5.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4951839, upper bound: 1.4932775
time: 5.05 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.44 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.44
Output dim: 8, lower bound: -1.4932782, upper bound: 1.4951834
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.44
Output dim: 8, lower bound: -1.4936696, upper bound: 1.4947928
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.44
Output dim: 8, lower bound: -1.4931797, upper bound: 1.4952815
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.44
Output dim: 8, lower bound: -1.4935711, upper bound: 1.4948901
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.44
Output dim: 8, lower bound: -1.4948905, upper bound: 1.4935701
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.44
Output dim: 8, lower bound: -1.4952819, upper bound: 1.4931788
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.44
Output dim: 8, lower bound: -1.4947925, upper bound: 1.4936689
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.44
Output dim: 8, lower bound: -1.4951839, upper bound: 1.4932775

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1060495, 3.1047211
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7919126, 2.7913003
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8374128, 2.8358455
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2954302, 3.2932177
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1592565, 2.1579223
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2899919, 3.2897587
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8427362, 2.8401694
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6719961, 2.6747046
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9938860, 2.9950705

Time for backsubstitution: 22.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5845

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4907750, upper bound: 1.4951763
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4932712, upper bound: 1.4926805
time: 4.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1063614, 3.1044087
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7922463, 2.7909670
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8377390, 2.8355203
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2956924, 3.2929554
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1581845, 2.1589937
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2892499, 3.2905011
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8431826, 2.8397241
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6726923, 2.6740086
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9945574, 2.9944000

Time for backsubstitution: 21.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5845

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4911666, upper bound: 1.4947851
time: 5.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4936626, upper bound: 1.4922892
time: 5.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1069136, 3.1038561
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7925858, 2.7906275
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8386545, 2.8346038
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2950420, 3.2936063
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1584992, 2.1586792
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2908053, 3.2889457
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8423796, 2.8405256
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6717834, 2.6749172
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9944277, 2.9945297

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 5845

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906773, upper bound: 1.4952753
time: 5.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4931727, upper bound: 1.4927782
time: 5.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1072254, 3.1035442
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7929196, 2.7902942
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8389807, 2.8342786
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2953043, 3.2933440
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1574273, 2.1597509
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2900624, 3.2896881
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8428259, 2.8400798
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6724796, 2.6742213
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9950972, 2.9938593

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 5845

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4910685, upper bound: 1.4948827
time: 5.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4935641, upper bound: 1.4923870
time: 5.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1035442, 3.1072254
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7902942, 2.7929192
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8342781, 2.8389802
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2933445, 3.2953038
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1597509, 2.1574275
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2896886, 3.2900624
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8400793, 2.8428259
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6742210, 2.6724796
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9938593, 2.9950972

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 5845

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4923875, upper bound: 1.4935631
time: 5.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4948835, upper bound: 1.4910683
time: 5.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1038561, 3.1069136
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7906280, 2.7925858
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8346043, 2.8386545
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2936068, 3.2950411
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1586795, 2.1584992
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2889457, 3.2908049
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8405256, 2.8423800
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6749172, 2.6717834
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9945297, 2.9944267

Time for backsubstitution: 22.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5845

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4927789, upper bound: 1.4931724
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952749, upper bound: 1.4906767
time: 4.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1044092, 3.1063614
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7909665, 2.7922468
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8355198, 2.8377385
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2929554, 3.2956924
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1589937, 2.1581845
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2905011, 3.2892494
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8397245, 2.8431821
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6740084, 2.6726923
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9944000, 2.9945564

Time for backsubstitution: 22.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5845

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4922895, upper bound: 1.4936615
time: 5.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4947858, upper bound: 1.4911659
time: 5.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1047211, 3.1060491
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7913003, 2.7919130
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8358459, 2.8374128
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2932186, 3.2954302
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1579223, 2.1592562
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2897592, 3.2899919
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8401690, 2.8427362
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6747046, 2.6719964
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9950705, 2.9938860

Time for backsubstitution: 22.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5845
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5845

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4926809, upper bound: 1.4932712
time: 6.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4951769, upper bound: 1.4907743
time: 5.56 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 34.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4907750, upper bound: 1.4951763
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4932712, upper bound: 1.4926805
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4911666, upper bound: 1.4947851
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4936626, upper bound: 1.4922892
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4906773, upper bound: 1.4952753
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4931727, upper bound: 1.4927782
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4910685, upper bound: 1.4948827
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4935641, upper bound: 1.4923870
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4923875, upper bound: 1.4935631
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4948835, upper bound: 1.4910683
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4927789, upper bound: 1.4931724
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4952749, upper bound: 1.4906767
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4922895, upper bound: 1.4936615
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4947858, upper bound: 1.4911659
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4926809, upper bound: 1.4932712
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.93
Output dim: 8, lower bound: -1.4951769, upper bound: 1.4907743

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1018047, 3.0986137
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7872930, 2.7880893
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8265810, 2.8202629
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2806044, 3.2829132
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1552949, 2.1551692
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2863955, 3.2845888
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8305178, 2.8316770
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6607165, 2.6668537
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9928522, 2.9935803

Time for backsubstitution: 22.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6253

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4907728, upper bound: 1.4933308
time: 5.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4889305, upper bound: 1.4951741
time: 4.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.0999422, 3.1004763
1: -9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.7887025, 2.7866807
2: -4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8218298, 2.8250136
3: -1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933
4: -14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2851257, 3.2783918
5: -8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1565032, 2.1539612
6: -12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2848220, 3.2861619
7: -9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8342438, 2.8279505
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6641450, 2.6634252
9: -7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9923964, 2.9940367

Time for backsubstitution: 22.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6253
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 5762

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6253

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4932690, upper bound: 1.4908352
time: 5.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4914260, upper bound: 1.4926781
time: 4.64 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 33.18 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 33.18
Output dim: 8, lower bound: -1.4907728, upper bound: 1.4933308
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 33.18
Output dim: 8, lower bound: -1.4889305, upper bound: 1.4951741
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 33.18
Output dim: 8, lower bound: -1.4932690, upper bound: 1.4908352
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 33.18
Output dim: 8, lower bound: -1.4914260, upper bound: 1.4926781
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4911666, upper bound: 1.4947851
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4936626, upper bound: 1.4922892
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4906773, upper bound: 1.4952753
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4931727, upper bound: 1.4927782
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4910685, upper bound: 1.4948827
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4935641, upper bound: 1.4923870
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4923875, upper bound: 1.4935631
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4948835, upper bound: 1.4910683
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4927789, upper bound: 1.4931724
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4952749, upper bound: 1.4906767
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4922895, upper bound: 1.4936615
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4947858, upper bound: 1.4911659
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4926809, upper bound: 1.4932712
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.18
Output dim: 8, lower bound: -1.4951769, upper bound: 1.4907743

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 60.35 + 545.47 = 605.83 seconds
