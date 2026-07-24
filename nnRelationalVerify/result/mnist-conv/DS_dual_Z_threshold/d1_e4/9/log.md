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
execution time: IAR + RelationalAnalysis = 22.24 + 33.92 = 56.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1784096, upper bound: 0.1784103

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 4667

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1784093, upper bound: 0.1778248
time: 5.20 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1778240, upper bound: 0.1784101
time: 3.90 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.27 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.27
Output dim: 5, lower bound: -0.1784093, upper bound: 0.1778248
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.27
Output dim: 5, lower bound: -0.1778240, upper bound: 0.1784101

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3974078, 0.3942533
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4864225, 0.4860020
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4717321, 0.4692264
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4205151, 0.4210854
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3834422, 0.3874962
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4589620, 0.4587195
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4174631, 0.4161391
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3382962, 0.3433867
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4498239, 0.4469662
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2586921, 0.2607646

Time for backsubstitution: 21.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 843

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 6123

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1784089, upper bound: 0.1767742
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1773588, upper bound: 0.1778244
time: 3.84 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3942533, 0.3974078
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4860020, 0.4864225
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4692264, 0.4717321
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4210854, 0.4205146
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3874962, 0.3834422
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4587197, 0.4589617
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4161391, 0.4174631
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3433867, 0.3382962
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4469662, 0.4498239
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2607646, 0.2586920

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 6123

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1778237, upper bound: 0.1773596
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1767734, upper bound: 0.1784097
time: 3.64 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.12 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.12
Output dim: 5, lower bound: -0.1784089, upper bound: 0.1767742
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.12
Output dim: 5, lower bound: -0.1773588, upper bound: 0.1778244
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.12
Output dim: 5, lower bound: -0.1778237, upper bound: 0.1773596
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.12
Output dim: 5, lower bound: -0.1767734, upper bound: 0.1784097

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

Time for backsubstitution: 21.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 843

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1784088, upper bound: 0.1764455
time: 5.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1780796, upper bound: 0.1767731
time: 6.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3964310, 0.3933983
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4742961, 0.4716372
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4727259, 0.4703641
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4111190, 0.4103463
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3835919, 0.3876677
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4524026, 0.4529819
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4093919, 0.4066601
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3313618, 0.3354313
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4479203, 0.4457297
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2534217, 0.2561517

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 843

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 843

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1773587, upper bound: 0.1774956
time: 4.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1770295, upper bound: 0.1778243
time: 4.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1

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

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 843

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1778235, upper bound: 0.1770303
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1774949, upper bound: 0.1773595
time: 3.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.3932765, 0.3965528
1: -13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4738755, 0.4720578
2: -8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4702201, 0.4728699
3: -4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4116898, 0.4097755
4: -8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.3876462, 0.3836133
5: 9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4521604, 0.4532242
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4080679, 0.4079840
7: -8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3364522, 0.3303409
8: -3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4450626, 0.4485874
9: -2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2554941, 0.2540792

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 843

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 843

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1767733, upper bound: 0.1780803
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1764447, upper bound: 0.1784095
time: 4.54 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.38 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 5, lower bound: -0.1784088, upper bound: 0.1764455
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 5, lower bound: -0.1780796, upper bound: 0.1767731
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.38
Output dim: 5, lower bound: -0.1773587, upper bound: 0.1774956
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 5, lower bound: -0.1770295, upper bound: 0.1778243
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 5, lower bound: -0.1778235, upper bound: 0.1770303
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.38
Output dim: 5, lower bound: -0.1774949, upper bound: 0.1773595
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 5, lower bound: -0.1767733, upper bound: 0.1780803
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.38
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

Time for backsubstitution: 21.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 174

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 1388

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1752401, upper bound: 0.1734050
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1752641, upper bound: 0.1733983
time: 4.22 seconds

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

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 174

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 1388

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1750174, upper bound: 0.1736389
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 407

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1745118, upper bound: 0.1744604
time: 7.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1757599, upper bound: 0.1732065
time: 3.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 174

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 1388

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1739782, upper bound: 0.1746805
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1739856, upper bound: 0.1746580
time: 3.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 174

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 1388

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1746573, upper bound: 0.1739856
time: 6.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1746799, upper bound: 0.1739788
time: 4.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 174

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 1388

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1736173, upper bound: 0.1750269
time: 5.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1736382, upper bound: 0.1750180
time: 4.21 seconds

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

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1388
type: DSZ, layer: 3, pos: 407
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1454
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 646
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1110
type: DSZ, layer: 3, pos: 1691
type: DSZ, layer: 3, pos: 1390
type: DSZ, layer: 3, pos: 2325
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 774
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 240
type: DSZ, layer: 3, pos: 3133
type: DSZ, layer: 3, pos: 1222
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 219
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2496
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 316
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2344
type: DSZ, layer: 3, pos: 1491
type: DSZ, layer: 3, pos: 2875
type: DSZ, layer: 3, pos: 174

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 3, pos: 1388

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1733977, upper bound: 0.1752647
time: 4.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1734044, upper bound: 0.1752407
time: 3.90 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.74
Output dim: 5, lower bound: -0.1752401, upper bound: 0.1734050
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.74
Output dim: 5, lower bound: -0.1752641, upper bound: 0.1733983
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.74
Output dim: 5, lower bound: -0.1745118, upper bound: 0.1744604
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.74
Output dim: 5, lower bound: -0.1757599, upper bound: 0.1732065
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.74
Output dim: 5, lower bound: -0.1739782, upper bound: 0.1746805
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.74
Output dim: 5, lower bound: -0.1739856, upper bound: 0.1746580
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.74
Output dim: 5, lower bound: -0.1746573, upper bound: 0.1739856
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.74
Output dim: 5, lower bound: -0.1746799, upper bound: 0.1739788
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.74
Output dim: 5, lower bound: -0.1736173, upper bound: 0.1750269
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.74
Output dim: 5, lower bound: -0.1736382, upper bound: 0.1750180
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.74
Output dim: 5, lower bound: -0.1733977, upper bound: 0.1752647
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.74
Output dim: 5, lower bound: -0.1734044, upper bound: 0.1752407

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 56.16 + 394.08 = 450.25 seconds
