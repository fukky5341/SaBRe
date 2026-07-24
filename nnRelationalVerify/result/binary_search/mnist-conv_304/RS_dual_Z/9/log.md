## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.0506091392
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.5822649, 3.5822649)
1: (-7.3978786, -4.1556597, -7.3978786, -4.1556597, -3.1666536, 3.1666532)
2: (-7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.9047523, 2.9047523)
3: (-11.2633400, -7.7441711, -11.2633400, -7.7441711, -3.4115591, 3.4115601)
4: (6.5621042, 8.8026104, 6.5621042, 8.8026104, -2.1841531, 2.1841531)
5: (-8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.9886804, 2.9886804)
6: (-12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.7548275, 3.7548275)
7: (-3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097)
8: (-6.9675961, -3.5078919, -6.9675961, -3.5078919, -3.2341232, 3.2341237)
9: (-5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.4804778, 2.4804778)

## BASE Result
execution time: IAR + LP analysis = 12.98 + 33.17 = 46.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.7321513, upper bound: 1.7321501


# Binary Search by BASE starts (time budget: 3553.85 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.8360369205474854
rel_dist={4: [-1.3410396013151953, 1.3410415304274936]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.6619789600372314
rel_dist={4: [-1.051660938425652, 1.051661396029865]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.545940637588501
rel_dist={4: [-0.792442823129953, 0.7924427706559456]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.603959560394287
rel_dist={4: [-0.9387819866503717, 0.9387850582514945]}

## Binary Search Result
Binary search time: 193.52 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3360.33 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4121141, upper bound: 1.4184886
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4184892, upper bound: 1.4121141
time: 6.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.49
Output dim: 4, lower bound: -1.4121141, upper bound: 1.4184886
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.49
Output dim: 4, lower bound: -1.4184892, upper bound: 1.4121141

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3563395, 3.3456488
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7456923, 2.7463875
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.6000371, 2.5806236
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9922390, 2.9855206
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8905890, 1.8969331
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6170487, 2.6036711
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5564709, 3.5591125
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7760611, 2.7594950
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2140789, 2.2203584

Time for backsubstitution: 13.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4121000, upper bound: 1.4171017
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4105803, upper bound: 1.4184761
time: 3.95 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3456488, 3.3514693
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7460709, 2.7456920
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5806236, 2.5912142
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9855204, 2.9891977
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8940566, 1.8905885
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6036716, 2.6109700
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5579138, 3.5564709
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7594948, 2.7685492
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2175126, 2.2140789

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4184756, upper bound: 1.4105805
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171014, upper bound: 1.4121001
time: 6.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.14
Output dim: 4, lower bound: -1.4121000, upper bound: 1.4171017
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.14
Output dim: 4, lower bound: -1.4105803, upper bound: 1.4184761
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.14
Output dim: 4, lower bound: -1.4184756, upper bound: 1.4105805
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.14
Output dim: 4, lower bound: -1.4171014, upper bound: 1.4121001

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3569331, 3.3453326
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7432804, 2.7508938
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.6000781, 2.5805986
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9917097, 2.9865088
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8925467, 1.8958857
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6155229, 2.6065183
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5548391, 3.5621576
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7700930, 2.7706571
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2174883, 2.2185283

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4120173, upper bound: 1.4103554
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4052611, upper bound: 1.4170025
time: 7.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3560233, 3.3456488
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7456923, 2.7439766
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.6000118, 2.5806236
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9922390, 2.9849913
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8895411, 1.8969331
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6170487, 2.6021452
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5564709, 3.5574808
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7760611, 2.7535272
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2122488, 2.2203584

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4105021, upper bound: 1.4117549
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4037810, upper bound: 1.4183563
time: 4.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3462424, 3.3511536
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7436609, 2.7501984
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5806651, 2.5911896
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9849920, 2.9901857
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8960142, 1.8895414
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6021457, 2.6138172
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5562820, 3.5595160
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7535276, 2.7797112
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2209215, 2.2122488

Time for backsubstitution: 13.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4183560, upper bound: 1.4037827
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4117544, upper bound: 1.4105021
time: 7.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3453326, 3.3514693
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7460709, 2.7432811
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5805984, 2.5912142
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9855204, 2.9886682
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8930092, 1.8905885
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6036716, 2.6094437
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5579138, 3.5548391
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7594948, 2.7625809
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2156825, 2.2140789

Time for backsubstitution: 13.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170020, upper bound: 1.4052614
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4103551, upper bound: 1.4120191
time: 5.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.44 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.44
Output dim: 4, lower bound: -1.4120173, upper bound: 1.4103554
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.44
Output dim: 4, lower bound: -1.4052611, upper bound: 1.4170025
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.44
Output dim: 4, lower bound: -1.4105021, upper bound: 1.4117549
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.44
Output dim: 4, lower bound: -1.4037810, upper bound: 1.4183563
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.44
Output dim: 4, lower bound: -1.4183560, upper bound: 1.4037827
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.44
Output dim: 4, lower bound: -1.4117544, upper bound: 1.4105021
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.44
Output dim: 4, lower bound: -1.4170020, upper bound: 1.4052614
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.44
Output dim: 4, lower bound: -1.4103551, upper bound: 1.4120191

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3573465, 3.3456759
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7478380, 2.7542989
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5999708, 2.5804596
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9847302, 2.9815617
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8970370, 1.8983653
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6139526, 2.6054065
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5615282, 3.5711036
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7741690, 2.7761095
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2190166, 2.2205732

Time for backsubstitution: 13.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3548024, upper bound: 1.3555239
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3548024, upper bound: 1.3555239
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3572769, 3.3457465
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7466869, 2.7554502
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5999393, 2.5804908
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9867625, 2.9795287
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8950267, 1.9003761
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6144114, 2.6049480
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5637846, 3.5688467
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7755461, 2.7747324
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2195334, 2.2200572

Time for backsubstitution: 13.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3479575, upper bound: 1.3622825
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3479575, upper bound: 1.3622825
time: 4.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3564377, 3.3459911
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7502489, 2.7473817
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5999041, 2.5804846
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9852576, 2.9800439
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8940320, 1.8994129
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6154790, 2.6010332
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5631590, 3.5664268
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7801371, 2.7589793
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2137780, 2.2224033

Time for backsubstitution: 13.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3548024, upper bound: 1.3555239
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3548024, upper bound: 1.3555239
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3563671, 3.3460617
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7490978, 2.7485330
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5998726, 2.5805159
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9872909, 2.9780111
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8920207, 1.9014237
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6159368, 2.6005747
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5654154, 3.5641699
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7815142, 2.7576025
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2142940, 2.2218874

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3479575, upper bound: 1.3622825
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3479575, upper bound: 1.3622825
time: 4.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3466558, 3.3514957
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7482166, 2.7536035
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5805578, 2.5910511
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9780107, 2.9852390
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.9005046, 1.8920205
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6005745, 2.6127050
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5629702, 3.5684614
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7576027, 2.7851632
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2224498, 2.2142942

Time for backsubstitution: 13.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3622809, upper bound: 1.3479580
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3622809, upper bound: 1.3479580
time: 4.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3465853, 3.3515663
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7470655, 2.7547548
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5805264, 2.5910823
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9800439, 2.9832063
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8984942, 1.8940313
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6010332, 2.6122465
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5652266, 3.5662050
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7589798, 2.7837861
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2229662, 2.2137778

Time for backsubstitution: 13.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3555223, upper bound: 1.3548024
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3555223, upper bound: 1.3548024
time: 4.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3457470, 3.3518124
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7506285, 2.7466862
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5804911, 2.5910757
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9785390, 2.9837217
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8974996, 1.8930683
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6021008, 2.6083317
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5646019, 3.5637846
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7635698, 2.7680330
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2172108, 2.2161243

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3622809, upper bound: 1.3479580
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3622809, upper bound: 1.3479580
time: 4.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3456764, 3.3518829
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7494764, 2.7478375
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5804596, 2.5911069
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9805713, 2.9816887
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8954883, 1.8950791
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6025596, 2.6078732
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5668592, 3.5615282
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7649469, 2.7666559
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2177267, 2.2156079

Time for backsubstitution: 13.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3555223, upper bound: 1.3548024
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3555223, upper bound: 1.3548024
time: 4.46 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3548024, upper bound: 1.3555239
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3548024, upper bound: 1.3555239
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3479575, upper bound: 1.3622825
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3479575, upper bound: 1.3622825
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3548024, upper bound: 1.3555239
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3548024, upper bound: 1.3555239
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3479575, upper bound: 1.3622825
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3479575, upper bound: 1.3622825
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3622809, upper bound: 1.3479580
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3622809, upper bound: 1.3479580
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3555223, upper bound: 1.3548024
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3555223, upper bound: 1.3548024
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3622809, upper bound: 1.3479580
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3622809, upper bound: 1.3479580
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3555223, upper bound: 1.3548024
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.10
Output dim: 4, lower bound: -1.3555223, upper bound: 1.3548024

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3551092, 3.3421309
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7415552, 2.7369080
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5962677, 2.5734360
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9809084, 2.9802155
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8964891, 1.8986254
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6275625, 2.5974624
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5524311, 3.5650878
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7697506, 2.7747571
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2177920, 2.2196438

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3486314, upper bound: 1.3406488
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3418639, upper bound: 1.3504697
time: 6.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3573465, 3.3434384
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7478380, 2.7480180
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5999708, 2.5767567
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9833832, 2.9815617
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8970370, 1.8978171
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6060085, 2.6054065
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5555134, 3.5711036
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7728167, 2.7761095
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2190166, 2.2193487

Time for backsubstitution: 13.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3486314, upper bound: 1.3406488
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3418639, upper bound: 1.3504697
time: 6.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3550386, 3.3422012
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7404051, 2.7380593
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5962362, 2.5734675
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9829407, 2.9781826
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8944783, 1.9006367
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6280212, 2.5970039
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5546875, 3.5628309
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7711277, 2.7733803
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2183080, 2.2191274

Time for backsubstitution: 13.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3417563, upper bound: 1.3475445
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3348283, upper bound: 1.3571924
time: 5.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3572769, 3.3435087
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7466869, 2.7491693
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5999393, 2.5767879
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9854164, 2.9795287
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8950267, 1.8998280
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6064672, 2.6049480
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5577698, 3.5688467
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7741938, 2.7747324
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2195334, 2.2188323

Time for backsubstitution: 13.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3417563, upper bound: 1.3475445
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3348283, upper bound: 1.3571926
time: 5.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3541994, 3.3436716
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7439671, 2.7356772
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5962014, 2.5743892
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9822817, 2.9786978
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8934836, 1.8998320
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6310930, 2.5930891
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5559330, 3.5604110
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7763157, 2.7576272
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2125530, 2.2218635

Time for backsubstitution: 13.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3486314, upper bound: 1.3406488
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3418639, upper bound: 1.3504697
time: 6.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3564377, 3.3437536
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7502489, 2.7411008
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5999041, 2.5767817
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9839115, 2.9800439
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8940320, 1.8988647
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6075349, 2.6010332
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5571442, 3.5664268
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7787848, 2.7589793
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2137780, 2.2211788

Time for backsubstitution: 13.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3486314, upper bound: 1.3406488
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3418639, upper bound: 1.3504697
time: 6.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3541298, 3.3437419
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7428160, 2.7368288
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5961699, 2.5744207
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9843149, 2.9766650
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8914728, 1.9018431
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6315517, 2.5926306
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5581894, 3.5581541
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7776928, 2.7562501
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2130694, 2.2213471

Time for backsubstitution: 13.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3417563, upper bound: 1.3475445
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3348283, upper bound: 1.3571924
time: 5.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3563671, 3.3438239
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7490978, 2.7422521
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5998726, 2.5768130
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9859447, 2.9780111
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8920207, 1.9008756
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6079936, 2.6005747
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5594006, 3.5641699
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7801619, 2.7576025
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2142940, 2.2206624

Time for backsubstitution: 13.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3417563, upper bound: 1.3475445
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3348283, upper bound: 1.3571926
time: 5.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3444185, 3.3479507
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7419357, 2.7362123
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5768547, 2.5840275
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9741898, 2.9838924
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8999567, 1.8922811
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6141853, 2.6047618
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5538731, 3.5624456
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7531843, 2.7838109
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2212253, 2.2133644

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3571924, upper bound: 1.3348288
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3475427, upper bound: 1.3417580
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3466558, 3.3492582
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7482166, 2.7473226
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5805578, 2.5873482
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9766655, 2.9852390
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.9005046, 1.8914728
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5926304, 2.6127050
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5569553, 3.5684614
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7562504, 2.7851632
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2224498, 2.2130692

Time for backsubstitution: 13.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3571924, upper bound: 1.3348288
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3475427, upper bound: 1.3417580
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3443480, 3.3480210
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7407846, 2.7373638
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5768232, 2.5840590
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9762230, 2.9818597
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8979454, 1.8942919
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6146431, 2.6043034
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5561314, 3.5601892
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7545614, 2.7824340
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2217417, 2.2128484

Time for backsubstitution: 13.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3504697, upper bound: 1.3418639
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3406471, upper bound: 1.3486314
time: 7.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3465853, 3.3493285
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7470655, 2.7484739
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5805264, 2.5873795
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9786978, 2.9832063
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8984942, 1.8934836
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5930891, 2.6122465
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5592117, 3.5662050
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7576265, 2.7837861
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2229662, 2.2125528

Time for backsubstitution: 13.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3504697, upper bound: 1.3418639
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3406471, upper bound: 1.3486331
time: 8.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3435087, 3.3494923
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7443457, 2.7349818
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5767884, 2.5849802
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9755621, 2.9823751
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8969507, 1.8934877
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6177158, 2.6003885
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5573750, 3.5577688
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7597504, 2.7666810
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2159863, 2.2155840

Time for backsubstitution: 13.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3571924, upper bound: 1.3348288
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3475427, upper bound: 1.3417580
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3457470, 3.3495748
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7506285, 2.7404053
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5804911, 2.5873725
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9771929, 2.9837217
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8974996, 1.8925202
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5941577, 2.6083317
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5585861, 3.5637846
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7622185, 2.7680330
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2172108, 2.2148988

Time for backsubstitution: 13.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3571924, upper bound: 1.3348288
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3475427, upper bound: 1.3417580
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3434381, 3.3495626
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7431946, 2.7361331
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5767570, 2.5850117
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9775953, 2.9803421
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8949404, 1.8954985
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6181746, 2.5999300
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5596333, 3.5555124
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7611275, 2.7653039
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2165022, 2.2150676

Time for backsubstitution: 13.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3504697, upper bound: 1.3418639
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3406471, upper bound: 1.3486314
time: 7.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3456764, 3.3496451
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7494764, 2.7415566
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5804596, 2.5874038
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9792261, 2.9816887
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8954883, 1.8945310
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5946155, 2.6078732
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5608425, 3.5615282
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7635956, 2.7666559
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2177267, 2.2143829

Time for backsubstitution: 13.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3504697, upper bound: 1.3418639
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3406471, upper bound: 1.3486331
time: 8.13 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3486314, upper bound: 1.3406488
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3418639, upper bound: 1.3504697
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3486314, upper bound: 1.3406488
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3418639, upper bound: 1.3504697
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3417563, upper bound: 1.3475445
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3348283, upper bound: 1.3571924
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3417563, upper bound: 1.3475445
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3348283, upper bound: 1.3571926
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3486314, upper bound: 1.3406488
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3418639, upper bound: 1.3504697
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3486314, upper bound: 1.3406488
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3418639, upper bound: 1.3504697
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3417563, upper bound: 1.3475445
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3348283, upper bound: 1.3571924
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3417563, upper bound: 1.3475445
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3348283, upper bound: 1.3571926
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3571924, upper bound: 1.3348288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3475427, upper bound: 1.3417580
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3571924, upper bound: 1.3348288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3475427, upper bound: 1.3417580
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3504697, upper bound: 1.3418639
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3406471, upper bound: 1.3486314
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3504697, upper bound: 1.3418639
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3406471, upper bound: 1.3486331
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3571924, upper bound: 1.3348288
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3475427, upper bound: 1.3417580
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3571924, upper bound: 1.3348288
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3475427, upper bound: 1.3417580
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3504697, upper bound: 1.3418639
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3406471, upper bound: 1.3486314
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3504697, upper bound: 1.3418639
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.82
Output dim: 4, lower bound: -1.3406471, upper bound: 1.3486331

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.2767935, 3.2594957
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.6940289, 2.6880183
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5764041, 2.5608101
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9401636, 2.9292135
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8960929, 1.8969588
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5953841, 2.5838528
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.4627547, 3.4532218
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7692389, 2.7702606
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2126486, 2.2113752

Time for backsubstitution: 12.26 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.8940565586090088
rel_dist={4: [-1.418532051192658, 1.418531738496264]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1483121, upper bound: 1.1597274
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597270, upper bound: 1.1483129
time: 4.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.53
Output dim: 4, lower bound: -1.1483121, upper bound: 1.1597274
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.53
Output dim: 4, lower bound: -1.1597270, upper bound: 1.1483129

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0860195, 3.0799110
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4933424, 2.4937401
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3786893, 2.3675964
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7359428, 2.7321033
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7165310, 1.7201560
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3643575, 2.3567133
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2619219, 3.2634315
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4561944, 2.4611955
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4896164, 2.4801505
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0562997, 2.0598879

Time for backsubstitution: 12.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1482936, upper bound: 1.1579149
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1464397, upper bound: 1.1597085
time: 10.18 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0799103, 3.0857315
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4937220, 2.4933426
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3675961, 2.3781869
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7321033, 2.7357805
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7199986, 1.7165306
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3567128, 2.3640118
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2633648, 3.2619219
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4609652, 2.4561949
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4801502, 2.4892046
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0597329, 2.0562997

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597083, upper bound: 1.1464396
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579157, upper bound: 1.1482931
time: 4.66 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.84 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.84
Output dim: 4, lower bound: -1.1482936, upper bound: 1.1579149
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.84
Output dim: 4, lower bound: -1.1464397, upper bound: 1.1597085
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.84
Output dim: 4, lower bound: -1.1597083, upper bound: 1.1464396
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.84
Output dim: 4, lower bound: -1.1579157, upper bound: 1.1482931

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0862231, 3.0795949
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4909315, 2.4952817
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3787026, 2.3675714
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7354136, 2.7324412
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7172003, 1.7191088
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3628316, 2.3576860
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2602901, 3.2644725
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4556098, 2.4615951
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4836493, 2.4839711
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0574636, 2.0580578

Time for backsubstitution: 12.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1482919, upper bound: 1.1540897
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1444784, upper bound: 1.1579143
time: 6.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0857043, 3.0799110
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4933424, 2.4913292
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3786645, 2.3675964
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7359428, 2.7315741
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7154832, 1.7201560
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3643575, 2.3551869
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2619219, 3.2617998
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4561944, 2.4606109
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4896164, 2.4741824
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0544701, 2.0598879

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1464379, upper bound: 1.1558793
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1426245, upper bound: 1.1597061
time: 6.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0801139, 3.0854158
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4913111, 2.4948843
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3676095, 2.3781624
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7315741, 2.7361181
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7206683, 1.7154834
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3551869, 2.3649845
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2617331, 3.2629628
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4603801, 2.4565945
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4741821, 2.4930251
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0608969, 2.0544696

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597061, upper bound: 1.1426269
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1558791, upper bound: 1.1464404
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0795951, 3.0857315
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4937220, 2.4909317
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3675714, 2.3781869
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7321033, 2.7352509
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7189512, 1.7165306
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3567128, 2.3624859
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2633648, 3.2602901
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4609652, 2.4556103
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4801502, 2.4832363
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0579033, 2.0562997

Time for backsubstitution: 12.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579135, upper bound: 1.1444779
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1540905, upper bound: 1.1482921
time: 6.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.30 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.30
Output dim: 4, lower bound: -1.1482919, upper bound: 1.1540897
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.30
Output dim: 4, lower bound: -1.1444784, upper bound: 1.1579143
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.30
Output dim: 4, lower bound: -1.1464379, upper bound: 1.1558793
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.30
Output dim: 4, lower bound: -1.1426245, upper bound: 1.1597061
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.30
Output dim: 4, lower bound: -1.1597061, upper bound: 1.1426269
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.30
Output dim: 4, lower bound: -1.1558791, upper bound: 1.1464404
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.30
Output dim: 4, lower bound: -1.1579135, upper bound: 1.1444779
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.30
Output dim: 4, lower bound: -1.1540905, upper bound: 1.1482921

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0866375, 3.0799685
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4949951, 2.4986868
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3785954, 2.3674457
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7284331, 2.7266226
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7208295, 1.7215879
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3612604, 2.3563776
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2669802, 3.2724514
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4558220, 2.4618478
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4877243, 2.4888332
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0589919, 2.0598819

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1243739, upper bound: 1.1308023
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1243739, upper bound: 1.1308028
time: 9.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0865974, 3.0800085
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4943371, 2.4993448
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3785772, 2.3674636
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7295947, 2.7254610
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7196803, 1.7227371
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3615227, 2.3561158
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2682695, 3.2711616
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4558630, 2.4618068
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4885111, 2.4880464
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0592875, 2.0595868

Time for backsubstitution: 12.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1205278, upper bound: 1.1345554
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1205278, upper bound: 1.1345554
time: 6.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0861177, 3.0802836
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4974060, 2.4947343
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3785572, 2.3674707
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7289605, 2.7257555
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7191119, 1.7226357
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3627877, 2.3538790
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2686110, 3.2697787
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4564071, 2.4608636
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4936924, 2.4790447
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0559983, 2.0617120

Time for backsubstitution: 12.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1243739, upper bound: 1.1309250
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1243739, upper bound: 1.1309255
time: 9.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0860777, 3.0803237
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4967489, 2.4953921
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3785391, 2.3674886
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7301221, 2.7245939
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7179627, 1.7237849
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3630490, 2.3536167
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2699003, 3.2684889
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4564476, 2.4608226
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4944792, 2.4782577
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0562935, 2.0614169

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1205278, upper bound: 1.1346918
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1205278, upper bound: 1.1346919
time: 9.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0805283, 3.0857882
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4953737, 2.4982893
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3675017, 2.3780372
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7245936, 2.7303002
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7242970, 1.7179625
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3536167, 2.3636761
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2684221, 3.2709417
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4605918, 2.4568472
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4782581, 2.4978869
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0624251, 2.0562932

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1346917, upper bound: 1.1205286
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1346917, upper bound: 1.1205286
time: 5.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0804882, 3.0858283
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4947166, 2.4989474
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3674841, 2.3780551
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7257552, 2.7291386
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7231479, 1.7191117
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3538790, 2.3634143
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2697115, 3.2696519
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4606328, 2.4568062
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4790449, 2.4970999
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0627203, 2.0559986

Time for backsubstitution: 12.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1309253, upper bound: 1.1243739
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1309253, upper bound: 1.1243738
time: 9.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0800085, 3.0861049
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4977846, 2.4943368
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3674641, 2.3780618
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7251220, 2.7294331
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7225795, 1.7190104
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3551431, 2.3611774
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2700529, 3.2682691
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4611769, 2.4558630
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4842253, 2.4880981
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0594316, 2.0581234

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1345553, upper bound: 1.1205288
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1345553, upper bound: 1.1205288
time: 7.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0799685, 3.0861449
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4971275, 2.4949946
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3674459, 2.3780797
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7262836, 2.7282715
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7214303, 1.7201593
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3554044, 2.3609152
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2713432, 3.2669792
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4612179, 2.4558220
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4850130, 2.4873114
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0597267, 2.0578287

Time for backsubstitution: 12.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1308025, upper bound: 1.1243741
time: 8.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1308025, upper bound: 1.1243742
time: 8.40 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 29.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1243739, upper bound: 1.1308023
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1243739, upper bound: 1.1308028
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1205278, upper bound: 1.1345554
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1205278, upper bound: 1.1345554
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1243739, upper bound: 1.1309250
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1243739, upper bound: 1.1309255
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1205278, upper bound: 1.1346918
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1205278, upper bound: 1.1346919
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1346917, upper bound: 1.1205286
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1346917, upper bound: 1.1205286
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1309253, upper bound: 1.1243739
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1309253, upper bound: 1.1243738
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1345553, upper bound: 1.1205288
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1345553, upper bound: 1.1205288
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1308025, upper bound: 1.1243741
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 4, lower bound: -1.1308025, upper bound: 1.1243742

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0844002, 3.0769837
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4887133, 2.4860573
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3748918, 2.3618453
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7256727, 2.7252765
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7202816, 1.7215023
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3656330, 2.3484335
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2592030, 3.2664356
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4556661, 2.4615595
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4846201, 2.4874811
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0577674, 2.0588257

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1166951, upper bound: 1.1114209
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1054462, upper bound: 1.1268351
time: 6.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0866375, 3.0777309
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4949951, 2.4924059
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3785954, 2.3637428
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7270870, 2.7266226
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7208295, 1.7210402
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3533173, 2.3563776
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2609634, 3.2724514
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4558220, 2.4616914
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4863720, 2.4888332
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0589919, 2.0586569

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1167475, upper bound: 1.1102223
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1073001, upper bound: 1.1268344
time: 5.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0843601, 3.0770237
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4880552, 2.4867153
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3748741, 2.3618631
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7268343, 2.7241149
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7191324, 1.7226510
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3658953, 2.3481717
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2604923, 3.2651458
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4557066, 2.4615188
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4854069, 2.4866941
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0580626, 2.0585306

Time for backsubstitution: 12.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1128202, upper bound: 1.1152842
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1016260, upper bound: 1.1306278
time: 6.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0865974, 3.0777709
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4943371, 2.4930639
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3785772, 2.3637607
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7282486, 2.7254610
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7196803, 1.7221894
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3535795, 2.3561158
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2622528, 3.2711616
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4558630, 2.4616504
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4871588, 2.4880464
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0592875, 2.0583618

Time for backsubstitution: 12.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1128831, upper bound: 1.1140416
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1034346, upper bound: 1.1306257
time: 7.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0838795, 3.0785244
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4911242, 2.4853542
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3748541, 2.3627985
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7270460, 2.7244093
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7185640, 1.7227087
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3691645, 2.3459349
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2627048, 3.2637630
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4562516, 2.4606619
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4911861, 2.4776924
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0547738, 2.0610449

Time for backsubstitution: 12.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1166951, upper bound: 1.1114209
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1054462, upper bound: 1.1268814
time: 6.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0861177, 3.0780461
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4974060, 2.4884534
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3785572, 2.3637679
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7276144, 2.7257555
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7191119, 1.7220876
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3548436, 2.3538790
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2625942, 3.2697787
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4564071, 2.4607072
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4923401, 2.4790447
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0559983, 2.0604866

Time for backsubstitution: 12.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1167475, upper bound: 1.1102223
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1073001, upper bound: 1.1268807
time: 5.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0838394, 3.0785644
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4904661, 2.4860120
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3748360, 2.3628163
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7282076, 2.7232478
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7174149, 1.7238576
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3694267, 2.3456726
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2639942, 3.2624731
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4562926, 2.4606209
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4919720, 2.4769056
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0550685, 2.0607498

Time for backsubstitution: 12.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1128202, upper bound: 1.1152842
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1016260, upper bound: 1.1306924
time: 6.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0860777, 3.0780861
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4967489, 2.4891112
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3785391, 2.3637857
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7287760, 2.7245939
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7179627, 1.7232368
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3551059, 2.3536167
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2638836, 3.2684889
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4564476, 2.4606662
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4931269, 2.4782577
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0562935, 2.0601919

Time for backsubstitution: 12.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1128831, upper bound: 1.1140416
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1034346, upper bound: 1.1306915
time: 7.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0782909, 3.0828035
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4890928, 2.4856598
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3637986, 2.3724368
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7218332, 2.7289536
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7237487, 1.7178764
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3579893, 2.3557329
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2606449, 3.2649260
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4604359, 2.4565589
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4751539, 2.4965348
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0612006, 2.0552375

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1306913, upper bound: 1.1034353
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1140408, upper bound: 1.1128833
time: 6.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0805283, 3.0835507
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4953737, 2.4920084
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3675017, 2.3743343
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7232475, 2.7303002
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7242970, 1.7174149
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3456726, 2.3636761
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2624073, 3.2709417
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4605918, 2.4566908
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4769049, 2.4978869
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0624251, 2.0550687

Time for backsubstitution: 12.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1306913, upper bound: 1.1016266
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1152833, upper bound: 1.1128202
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0782509, 3.0828435
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4884348, 2.4863179
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3637810, 2.3724546
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7229948, 2.7277920
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7225995, 1.7190256
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3582516, 2.3554711
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2619362, 3.2636361
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4604769, 2.4565182
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4759407, 2.4957478
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0614958, 2.0549428

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1268811, upper bound: 1.1073008
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1102200, upper bound: 1.1167476
time: 7.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0804882, 3.0835907
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4947166, 2.4926665
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3674841, 2.3743522
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7244091, 2.7291386
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7231479, 1.7185636
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3459349, 2.3634143
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2636967, 3.2696519
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4606328, 2.4566498
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4776926, 2.4970999
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0627203, 2.0547740

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1268811, upper bound: 1.1054467
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1114184, upper bound: 1.1166950
time: 4.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0777712, 3.0843451
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4915028, 2.4849567
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3637609, 2.3733895
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7232065, 2.7280865
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7220311, 1.7190831
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3615208, 2.3532343
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2641468, 3.2622533
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4610214, 2.4556613
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4817200, 2.4867461
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0582070, 2.0574567

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1306256, upper bound: 1.1034345
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1140408, upper bound: 1.1128833
time: 6.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0800085, 3.0838673
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4977846, 2.4880559
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3674641, 2.3743587
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7237759, 2.7294331
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7225795, 1.7184622
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3471990, 2.3611774
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2640381, 3.2682691
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4611769, 2.4557066
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4828739, 2.4880981
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0594316, 2.0568988

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1306256, upper bound: 1.1016257
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1152833, upper bound: 1.1128202
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0777311, 3.0843852
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4908447, 2.4856145
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3637428, 2.3734074
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7243681, 2.7269249
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7208824, 1.7202322
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3617821, 2.3529720
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2654381, 3.2609634
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4610624, 2.4556203
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4825058, 2.4859593
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0585022, 2.0571620

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1268350, upper bound: 1.1073002
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1102200, upper bound: 1.1167476
time: 7.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0799685, 3.0839074
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4971275, 2.4887137
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3674459, 2.3743765
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7249374, 2.7282715
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7214303, 1.7196112
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3474612, 2.3609152
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2653275, 3.2669792
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4612179, 2.4556656
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4836607, 2.4873114
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0597267, 2.0566037

Time for backsubstitution: 12.27 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.7199985980987549
rel_dist={4: [-1.1597516071645249, 1.1597543492710418]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0422059, upper bound: 1.0516570
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516541, upper bound: 1.0422068
time: 5.82 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.12 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.12
Output dim: 4, lower bound: -1.0422059, upper bound: 1.0516570
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.12
Output dim: 4, lower bound: -1.0516541, upper bound: 1.0422068

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -2.9959135, 2.9913316
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4092264, 2.4095242
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3049073, 2.2965872
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6505098, 2.6476309
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6585114, 1.6612303
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2801270, 2.2743940
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1637392, 3.1648717
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3818426, 2.3855934
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.3941355, 2.3870356
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0037065, 2.0063977

Time for backsubstitution: 13.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0421883, upper bound: 1.0497427
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0402885, upper bound: 1.0516395
time: 6.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -2.9913311, 2.9959135
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4095240, 2.4092259
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.2965875, 2.3049071
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6476307, 2.6505103
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6612303, 1.6585112
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2743936, 2.2801270
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1648712, 3.1637392
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3855929, 2.3818429
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.3870354, 2.3941355
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0063977, 2.0037065

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516366, upper bound: 1.0402897
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0497398, upper bound: 1.0421887
time: 11.12 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 31.40 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 31.40
Output dim: 4, lower bound: -1.0421883, upper bound: 1.0497427
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 31.40
Output dim: 4, lower bound: -1.0402885, upper bound: 1.0516395
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 31.40
Output dim: 4, lower bound: -1.0516366, upper bound: 1.0402897
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 31.40
Output dim: 4, lower bound: -1.0497398, upper bound: 1.0421887

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -2.9955974, 2.9913316
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4092264, 2.4071133
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3048820, 2.2965872
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6505098, 2.6471016
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6574640, 1.6612303
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2801270, 2.2728677
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1637392, 3.1632400
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3818426, 2.3850088
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.3941355, 2.3810675
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0018768, 2.0063977

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0402870, upper bound: 1.0487449
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0373954, upper bound: 1.0516380
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -2.9914050, 2.9955974
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4071131, 2.4097795
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.2965908, 2.3048820
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6471014, 2.6506314
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6614709, 1.6574638
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2728677, 2.2804751
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1632395, 3.1641121
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3850083, 2.3819964
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.3810673, 2.3955088
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0068130, 2.0018764

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516351, upper bound: 1.0373963
time: 7.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0487421, upper bound: 1.0402880
time: 8.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 31.08 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 31.08
Output dim: 4, lower bound: -1.0402870, upper bound: 1.0487449
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.08
Output dim: 4, lower bound: -1.0373954, upper bound: 1.0516380
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.08
Output dim: 4, lower bound: -1.0516351, upper bound: 1.0373963
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 31.08
Output dim: 4, lower bound: -1.0487421, upper bound: 1.0402880

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -2.9959812, 2.9917445
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4126320, 2.4110117
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3047609, 2.2964795
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6444001, 2.6401215
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6599436, 1.6645718
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2787538, 2.2712975
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1713953, 3.1699290
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3820858, 2.3852208
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.3988008, 2.3851428
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0036263, 2.0079267

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.34 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0193723, upper bound: 1.0352967
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0221417, upper bound: 1.0315756
time: 6.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -2.9918194, 2.9959810
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4110126, 2.4131846
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.2964830, 2.3047609
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6401219, 2.6445224
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6648126, 1.6599433
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2712975, 2.2791014
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1699295, 3.1717682
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3852205, 2.3822391
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.3851433, 2.4001744
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0083418, 2.0036266

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0315747, upper bound: 1.0221419
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0352937, upper bound: 1.0193726
time: 7.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 28.27 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.27
Output dim: 4, lower bound: -1.0193723, upper bound: 1.0352967
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 28.27
Output dim: 4, lower bound: -1.0221417, upper bound: 1.0315756
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.27
Output dim: 4, lower bound: -1.0315747, upper bound: 1.0221419
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 28.27
Output dim: 4, lower bound: -1.0352937, upper bound: 1.0193726
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.6619789600372314
rel_dist={4: [-1.051660938425652, 1.051661396029865]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1848.43 seconds
