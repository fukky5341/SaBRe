## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.165768669


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5442555, 0.5442555)
1: (-6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3568206, 0.3568206)
2: (-7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4619796, 0.4619796)
3: (-2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4880099, 0.4880099)
4: (-5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5492563, 0.5492564)
5: (-9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4988909, 0.4988909)
6: (-15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3130085, 0.3130085)
7: (4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2986627, 0.2986629)
8: (-5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4199564, 0.4199564)
9: (-3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3778250, 0.3778253)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.68 + 32.96 = 55.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1674431, upper bound: 0.1674431

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 5792
type: DSZ, layer: 1, pos: 5846

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 117

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1656787, upper bound: 0.1674425
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674427, upper bound: 0.1656786
time: 4.45 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.46 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.46
Output dim: 7, lower bound: -0.1656787, upper bound: 0.1674425
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.46
Output dim: 7, lower bound: -0.1674427, upper bound: 0.1656786

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5442555, 0.5442572
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3568206, 0.3568202
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4619799, 0.4619794
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4880087, 0.4880099
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5492563, 0.5492554
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4988904, 0.4988909
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3130085, 0.3130092
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2986624, 0.2986629
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4199557, 0.4199556
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3778260, 0.3778253

Time for backsubstitution: 21.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 5792
type: DSZ, layer: 1, pos: 5846

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 522

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1655467, upper bound: 0.1674424
time: 6.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655971, upper bound: 0.1655473
time: 4.21 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5442574, 0.5442553
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3568201, 0.3568206
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4619794, 0.4619799
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4880102, 0.4880087
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5492554, 0.5492563
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4988909, 0.4988904
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3130093, 0.3130085
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2986629, 0.2986624
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4199557, 0.4199556
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3778250, 0.3778259

Time for backsubstitution: 21.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 5792
type: DSZ, layer: 1, pos: 5846

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 522

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655475, upper bound: 0.1655970
time: 5.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674426, upper bound: 0.1655465
time: 5.18 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 32.59 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 32.59
Output dim: 7, lower bound: -0.1655467, upper bound: 0.1674424
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 32.59
Output dim: 7, lower bound: -0.1655971, upper bound: 0.1655473
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 32.59
Output dim: 7, lower bound: -0.1655475, upper bound: 0.1655970
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 32.59
Output dim: 7, lower bound: -0.1674426, upper bound: 0.1655465

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5436959, 0.5436180
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3480501, 0.3467162
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4206808, 0.4147737
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4696150, 0.4719391
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5367489, 0.5349610
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4471605, 0.4537463
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3081098, 0.3074106
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2832565, 0.2852207
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4165593, 0.4170532
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3590968, 0.3614802

Time for backsubstitution: 22.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5792
type: DSZ, layer: 1, pos: 5846

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5792

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1655465, upper bound: 0.1670945
time: 5.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1651988, upper bound: 0.1674423
time: 3.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5436182, 0.5436959
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3467162, 0.3480500
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4147737, 0.4206808
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4719391, 0.4696150
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5349610, 0.5367489
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4537463, 0.4471605
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3074106, 0.3081098
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2852209, 0.2832566
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4170533, 0.4165595
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3614805, 0.3590964

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5792
type: DSZ, layer: 1, pos: 5846

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 5792

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674423, upper bound: 0.1651986
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1670946, upper bound: 0.1655464
time: 2.99 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.60 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.60
Output dim: 7, lower bound: -0.1655465, upper bound: 0.1670945
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.60
Output dim: 7, lower bound: -0.1651988, upper bound: 0.1674423
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.60
Output dim: 7, lower bound: -0.1674423, upper bound: 0.1651986
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.60
Output dim: 7, lower bound: -0.1670946, upper bound: 0.1655464

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5057352, 0.5002397
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3420951, 0.3415028
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4011378, 0.3924274
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4426711, 0.4483550
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5068488, 0.5007952
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4197457, 0.4297585
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3070006, 0.3064833
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2816164, 0.2833464
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4081714, 0.4097111
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3352927, 0.3342863

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5846

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 5846

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1628983, upper bound: 0.1670909
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655428, upper bound: 0.1644463
time: 3.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5003178, 0.5056572
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3428366, 0.3407613
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3983345, 0.3952310
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4460309, 0.4449952
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5025835, 0.5050606
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4231725, 0.4263315
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3071824, 0.3063014
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2813821, 0.2835805
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4092174, 0.4086653
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3319024, 0.3376766

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5846

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 5846

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1625506, upper bound: 0.1674385
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1651951, upper bound: 0.1647940
time: 3.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5056572, 0.5003177
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3407612, 0.3428366
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3952308, 0.3983345
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4449952, 0.4460309
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5050607, 0.5025834
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4263315, 0.4231725
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3063014, 0.3071824
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2835805, 0.2813821
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4086654, 0.4092174
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3376766, 0.3319024

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5846

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 5846

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1647942, upper bound: 0.1651949
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674387, upper bound: 0.1625504
time: 3.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5002396, 0.5057352
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3415027, 0.3420951
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3924274, 0.4011378
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4483552, 0.4426711
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5007954, 0.5068487
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4297585, 0.4197457
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3064832, 0.3070006
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2833464, 0.2816162
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4097111, 0.4081715
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3342863, 0.3352927

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5846

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 5846

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1644465, upper bound: 0.1655426
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1670910, upper bound: 0.1628981
time: 3.04 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.98 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 7, lower bound: -0.1628983, upper bound: 0.1670909
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.98
Output dim: 7, lower bound: -0.1655428, upper bound: 0.1644463
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 7, lower bound: -0.1625506, upper bound: 0.1674385
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.98
Output dim: 7, lower bound: -0.1651951, upper bound: 0.1647940
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.98
Output dim: 7, lower bound: -0.1647942, upper bound: 0.1651949
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 7, lower bound: -0.1674387, upper bound: 0.1625504
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.98
Output dim: 7, lower bound: -0.1644465, upper bound: 0.1655426
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 7, lower bound: -0.1670910, upper bound: 0.1628981

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4996864, 0.4933279
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3396292, 0.3386852
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4017596, 0.3933756
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4160213, 0.4176428
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4848473, 0.4816585
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.3997793, 0.4068534
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2996830, 0.2981209
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2672338, 0.2707629
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3988495, 0.3987573
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3243340, 0.3247476

Time for backsubstitution: 21.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3102

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1623106, upper bound: 0.1670374
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1628447, upper bound: 0.1665033
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4942690, 0.4987453
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3403707, 0.3379437
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3989563, 0.3961792
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4193811, 0.4142830
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4805818, 0.4859238
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4032063, 0.4034266
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2998649, 0.2979390
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2669997, 0.2709970
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3998957, 0.3977115
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3209437, 0.3281379

Time for backsubstitution: 21.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3102

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 3, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1619629, upper bound: 0.1673850
time: 4.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1624970, upper bound: 0.1668510
time: 3.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4987453, 0.4942690
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3379436, 0.3403707
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3961792, 0.3989563
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4142830, 0.4193811
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4859238, 0.4805818
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4034266, 0.4032063
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2979391, 0.2998649
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2709970, 0.2669997
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3977118, 0.3998955
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3281378, 0.3209437

Time for backsubstitution: 21.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3102

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1668510, upper bound: 0.1624968
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1673851, upper bound: 0.1619627
time: 3.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4933277, 0.4996864
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3386850, 0.3396292
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3933756, 0.4017596
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4176428, 0.4160213
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4816585, 0.4848473
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4068537, 0.3997796
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2981209, 0.2996830
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2707629, 0.2672341
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3987575, 0.3988497
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3247474, 0.3243340

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3102

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1665033, upper bound: 0.1628445
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1670374, upper bound: 0.1623104
time: 3.11 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.50
Output dim: 7, lower bound: -0.1623106, upper bound: 0.1670374
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.50
Output dim: 7, lower bound: -0.1628447, upper bound: 0.1665033
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.50
Output dim: 7, lower bound: -0.1619629, upper bound: 0.1673850
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.50
Output dim: 7, lower bound: -0.1624970, upper bound: 0.1668510
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.50
Output dim: 7, lower bound: -0.1668510, upper bound: 0.1624968
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.50
Output dim: 7, lower bound: -0.1673851, upper bound: 0.1619627
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.50
Output dim: 7, lower bound: -0.1665033, upper bound: 0.1628445
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.50
Output dim: 7, lower bound: -0.1670374, upper bound: 0.1623104

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4928246, 0.4860998
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3180225, 0.3177009
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3991973, 0.3917122
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4164164, 0.4181695
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4819446, 0.4789083
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.3819404, 0.3894026
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2992837, 0.2977359
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2663916, 0.2699692
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3837643, 0.3817034
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3112097, 0.3108987

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3102

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 310

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1610933, upper bound: 0.1659880
time: 4.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1612614, upper bound: 0.1658201
time: 3.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4924583, 0.4864662
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3186449, 0.3170785
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4000964, 0.3908131
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4165478, 0.4180379
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4820971, 0.4787555
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.3823287, 0.3890145
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2992980, 0.2977216
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2664403, 0.2699206
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3817956, 0.3836719
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3104854, 0.3116230

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3102

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 310

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1616274, upper bound: 0.1654539
time: 4.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1617955, upper bound: 0.1652860
time: 3.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4874072, 0.4915173
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3187640, 0.3169594
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3963938, 0.3945158
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4197762, 0.4148097
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4776793, 0.4831736
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.3853672, 0.3859760
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2994657, 0.2975540
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2661575, 0.2702034
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3848100, 0.3806576
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3078191, 0.3142890

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3102

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 310

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1607457, upper bound: 0.1663358
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1609137, upper bound: 0.1661678
time: 3.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4870410, 0.4918835
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3193864, 0.3163370
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3972929, 0.3936167
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4199076, 0.4146781
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4778318, 0.4830210
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.3857553, 0.3855876
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2994798, 0.2975397
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2662061, 0.2701547
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3828416, 0.3826261
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3070951, 0.3150133

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3102

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 310

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1612798, upper bound: 0.1658017
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1614478, upper bound: 0.1656337
time: 2.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4918835, 0.4870410
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3163370, 0.3193864
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3936167, 0.3972929
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4146781, 0.4199076
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4830208, 0.4778316
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.3855875, 0.3857553
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2975397, 0.2994798
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2701548, 0.2662063
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3826261, 0.3828416
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3150134, 0.3070949

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3102

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 310

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1656337, upper bound: 0.1614477
time: 4.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1658017, upper bound: 0.1612797
time: 3.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4915173, 0.4874072
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3169594, 0.3187640
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3945158, 0.3963938
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4148097, 0.4197762
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4831738, 0.4776790
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.3859761, 0.3853672
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2975540, 0.2994656
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2702035, 0.2661574
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3806577, 0.3848101
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3142891, 0.3078191

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 3102

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 310

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1661678, upper bound: 0.1609135
time: 4.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1663358, upper bound: 0.1607456
time: 3.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4864662, 0.4924583
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3170785, 0.3186449
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3908131, 0.4000964
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4180379, 0.4165478
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4787555, 0.4820971
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.3890145, 0.3823287
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2977216, 0.2992980
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2699205, 0.2664404
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3836720, 0.3817958
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3116231, 0.3104852

Time for backsubstitution: 22.13 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.63 + 560.76 = 616.39 seconds
