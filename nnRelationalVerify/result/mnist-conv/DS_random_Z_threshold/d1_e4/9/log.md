## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.177517552


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.4194586, 0.4194586)
1: (-13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4893668, 0.4893668)
2: (-8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4813509, 0.4813509)
3: (-4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4216642, 0.4216642)
4: (-8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.4117608, 0.4117606)
5: (9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4606638, 0.4606638)
6: (-11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4267104, 0.4267104)
7: (-8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3734529, 0.3734527)
8: (-3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4663706, 0.4663701)
9: (-2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2750261, 0.2750260)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.35 + 33.28 = 56.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1784096, upper bound: 0.1784103

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 4667

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6123

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1784092, upper bound: 0.1773590
time: 6.71 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1773591, upper bound: 0.1784099
time: 3.78 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.51 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.51
Output dim: 5, lower bound: -0.1784092, upper bound: 0.1773590
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.51
Output dim: 5, lower bound: -0.1773591, upper bound: 0.1784099

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.4186037, 0.4184818
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4750009, 0.4772394
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4824886, 0.4823446
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4109247, 0.4122684
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.4119325, 0.4119112
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4549274, 0.4541054
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4172316, 0.4186394
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3654974, 0.3665180
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4651346, 0.4644680
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2704135, 0.2697561

Time for backsubstitution: 21.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 843

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4667

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1784089, upper bound: 0.1767742
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1778237, upper bound: 0.1773596
time: 3.42 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.4184818, 0.4186037
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4772394, 0.4750009
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4823446, 0.4824886
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4122684, 0.4109247
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.4119110, 0.4119325
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4541054, 0.4549274
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4186397, 0.4172316
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3665180, 0.3654974
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4644680, 0.4651346
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2697560, 0.2704136

Time for backsubstitution: 21.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 4667

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 843

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1773590, upper bound: 0.1780806
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1770298, upper bound: 0.1784098
time: 3.99 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.09 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.09
Output dim: 5, lower bound: -0.1784089, upper bound: 0.1767742
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.09
Output dim: 5, lower bound: -0.1778237, upper bound: 0.1773596
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.09
Output dim: 5, lower bound: -0.1773590, upper bound: 0.1780806
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.09
Output dim: 5, lower bound: -0.1770298, upper bound: 0.1784098

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3965528, 0.3932765
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4720578, 0.4738755
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4728699, 0.4702201
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4097753, 0.4116900
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3836133, 0.3876462
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4532247, 0.4521604
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4079840, 0.4080679
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3303409, 0.3364522
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4485874, 0.4450626
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2540792, 0.2554941

Time for backsubstitution: 21.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 843

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 843

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1784088, upper bound: 0.1764455
time: 5.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1780796, upper bound: 0.1767731
time: 5.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3933983, 0.3964310
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4716372, 0.4742961
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4703641, 0.4727259
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4103465, 0.4111192
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3876677, 0.3835919
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4529819, 0.4524026
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4066601, 0.4093919
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3354313, 0.3313615
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4457297, 0.4479203
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2561517, 0.2534217

Time for backsubstitution: 21.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 843

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 843

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1778235, upper bound: 0.1770303
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1774949, upper bound: 0.1773595
time: 3.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.4144824, 0.4151042
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4783759, 0.4765685
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4777946, 0.4785075
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4080496, 0.4072332
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.4119539, 0.4119844
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4540782, 0.4548957
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4168422, 0.4156594
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3615594, 0.3613636
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4644938, 0.4651661
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2658408, 0.2659390

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4667

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4667

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1773587, upper bound: 0.1774956
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1767733, upper bound: 0.1780803
time: 3.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.4149823, 0.4146042
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4788067, 0.4761374
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4783635, 0.4779387
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4085770, 0.4067059
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.4119630, 0.4119754
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4540734, 0.4548995
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4170673, 0.4154344
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3623846, 0.3605387
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4644995, 0.4651608
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2652814, 0.2664983

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4667

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4667

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1770295, upper bound: 0.1778243
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1764447, upper bound: 0.1784095
time: 4.47 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.69 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 5, lower bound: -0.1784088, upper bound: 0.1764455
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 5, lower bound: -0.1780796, upper bound: 0.1767731
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 5, lower bound: -0.1778235, upper bound: 0.1770303
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.69
Output dim: 5, lower bound: -0.1774949, upper bound: 0.1773595
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.69
Output dim: 5, lower bound: -0.1773587, upper bound: 0.1774956
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 5, lower bound: -0.1767733, upper bound: 0.1780803
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 5, lower bound: -0.1770295, upper bound: 0.1778243
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 5, lower bound: -0.1764447, upper bound: 0.1784095

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3925531, 0.3897767
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4731939, 0.4754424
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4683194, 0.4662385
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4055562, 0.4079981
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3836567, 0.3876984
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4531975, 0.4521294
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4061871, 0.4064960
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3253820, 0.3323185
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4486141, 0.4450951
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2501645, 0.2510202

Time for backsubstitution: 21.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 1388

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1685

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1778014, upper bound: 0.1764445
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1784078, upper bound: 0.1758430
time: 3.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3930531, 0.3892767
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4736247, 0.4750116
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4688883, 0.4656701
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4060836, 0.4074707
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3836656, 0.3876894
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4531937, 0.4521337
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4064121, 0.4062710
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3262072, 0.3314934
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4486194, 0.4450893
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2496053, 0.2515795

Time for backsubstitution: 20.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 3102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1390

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1763512, upper bound: 0.1751454
time: 3.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1764506, upper bound: 0.1750464
time: 4.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3893986, 0.3929312
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4727733, 0.4758630
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4658136, 0.4687443
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4061270, 0.4074273
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3877108, 0.3836441
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4529552, 0.4523716
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4048631, 0.4078200
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3304727, 0.3272281
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4457564, 0.4479523
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2522371, 0.2489477

Time for backsubstitution: 21.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 1691

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1390

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1760959, upper bound: 0.1754013
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1761948, upper bound: 0.1753022
time: 3.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3892767, 0.3930531
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4750116, 0.4736247
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4656696, 0.4688883
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4074707, 0.4060836
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3876894, 0.3836656
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4521332, 0.4531937
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4062710, 0.4064121
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3314934, 0.3262072
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4450893, 0.4486194
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2515795, 0.2496053

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 2325

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 963

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766625, upper bound: 0.1760584
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1747635, upper bound: 0.1779608
time: 3.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3929312, 0.3893986
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4758630, 0.4727731
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4687443, 0.4658136
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4074273, 0.4061272
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3836441, 0.3877108
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4523716, 0.4529552
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4078200, 0.4048631
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3272281, 0.3304727
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4479523, 0.4457564
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2489477, 0.2522371

Time for backsubstitution: 21.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 1390

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3133

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766556, upper bound: 0.1774436
time: 5.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766538, upper bound: 0.1774460
time: 3.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3897767, 0.3925531
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4754424, 0.4731936
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4662385, 0.4683194
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4079981, 0.4055562
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3876984, 0.3836565
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4521294, 0.4531975
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4064960, 0.4061871
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3323185, 0.3253820
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4450951, 0.4486141
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2510202, 0.2501645

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 3109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2344

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1764441, upper bound: 0.1778536
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1758949, upper bound: 0.1784089
time: 3.73 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.39 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 5, lower bound: -0.1778014, upper bound: 0.1764445
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 5, lower bound: -0.1784078, upper bound: 0.1758430
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.39
Output dim: 5, lower bound: -0.1763512, upper bound: 0.1751454
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.39
Output dim: 5, lower bound: -0.1764506, upper bound: 0.1750464
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.39
Output dim: 5, lower bound: -0.1760959, upper bound: 0.1754013
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.39
Output dim: 5, lower bound: -0.1761948, upper bound: 0.1753022
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.39
Output dim: 5, lower bound: -0.1766625, upper bound: 0.1760584
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 5, lower bound: -0.1747635, upper bound: 0.1779608
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.39
Output dim: 5, lower bound: -0.1766556, upper bound: 0.1774436
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.39
Output dim: 5, lower bound: -0.1766538, upper bound: 0.1774460
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 5, lower bound: -0.1764441, upper bound: 0.1778536
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 5, lower bound: -0.1758949, upper bound: 0.1784089

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3936884, 0.3914897
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4787483, 0.4810376
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4710441, 0.4679337
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4020932, 0.4038563
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3749187, 0.3790059
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4535007, 0.4525054
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4064348, 0.4067116
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3235872, 0.3300478
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4484506, 0.4449325
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2455243, 0.2460133

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1110

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 403

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1770064, upper bound: 0.1750138
time: 6.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1763634, upper bound: 0.1756526
time: 3.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3942659, 0.3909121
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4787889, 0.4809971
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4700146, 0.4689631
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4014142, 0.4045351
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3749642, 0.3789604
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4535732, 0.4524329
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4064026, 0.4067435
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3231113, 0.3305237
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4484515, 0.4449315
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2451576, 0.2463800

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 3133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2376

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1783907, upper bound: 0.1735438
time: 4.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1761089, upper bound: 0.1758258
time: 5.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3891566, 0.3929133
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4749997, 0.4735115
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4655800, 0.4687257
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4073229, 0.4059904
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3875518, 0.3834567
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4521065, 0.4531741
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4061852, 0.4063239
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3314824, 0.3261883
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4450598, 0.4485793
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2514941, 0.2494725

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 407

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 174

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1738564, upper bound: 0.1778545
time: 6.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1746252, upper bound: 0.1770895
time: 4.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3897736, 0.3925507
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4754398, 0.4731901
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4662318, 0.4683146
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4079967, 0.4055536
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3876991, 0.3836567
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4521289, 0.4531965
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4064894, 0.4061823
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3323188, 0.3253822
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4450946, 0.4486141
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2510203, 0.2501645

Time for backsubstitution: 22.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 2376

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1222

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1760646, upper bound: 0.1770099
time: 5.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1755982, upper bound: 0.1775048
time: 6.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3897746, 0.3925498
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4754386, 0.4731910
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4662337, 0.4683123
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4079957, 0.4055545
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3876984, 0.3836575
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4521284, 0.4531970
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4064913, 0.4061804
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3323188, 0.3253822
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4450951, 0.4486136
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2510201, 0.2501647

Time for backsubstitution: 22.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 3133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1491

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1758948, upper bound: 0.1764945
time: 5.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1739791, upper bound: 0.1784087
time: 4.91 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 32.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.99
Output dim: 5, lower bound: -0.1770064, upper bound: 0.1750138
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.99
Output dim: 5, lower bound: -0.1763634, upper bound: 0.1756526
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.99
Output dim: 5, lower bound: -0.1783907, upper bound: 0.1735438
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.99
Output dim: 5, lower bound: -0.1761089, upper bound: 0.1758258
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.99
Output dim: 5, lower bound: -0.1738564, upper bound: 0.1778545
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.99
Output dim: 5, lower bound: -0.1746252, upper bound: 0.1770895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.99
Output dim: 5, lower bound: -0.1760646, upper bound: 0.1770099
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.99
Output dim: 5, lower bound: -0.1755982, upper bound: 0.1775048
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.99
Output dim: 5, lower bound: -0.1758948, upper bound: 0.1764945
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.99
Output dim: 5, lower bound: -0.1739791, upper bound: 0.1784087

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3924160, 0.3896148
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4655585, 0.4654717
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4633179, 0.4607644
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4055820, 0.4084365
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3759592, 0.3797102
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4530077, 0.4518962
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4093790, 0.4097741
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3027298, 0.3097095
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4480829, 0.4444489
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2466444, 0.2478487

Time for backsubstitution: 22.42 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.62 + 553.95 = 610.58 seconds
