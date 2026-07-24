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
execution time: IAR + LP analysis = 14.71 + 33.36 = 48.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.7321513, upper bound: 1.7321501


# Binary Search by BASE starts (time budget: 3551.93 seconds, max iter: 100)

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
Binary search time: 195.16 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3356.77 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4185187, upper bound: 1.4171448
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171447, upper bound: 1.4185205
time: 6.25 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.85 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.85
Output dim: 4, lower bound: -1.4185187, upper bound: 1.4171448
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.85
Output dim: 4, lower bound: -1.4171447, upper bound: 1.4185205

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3520632, 3.3511536
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7436609, 2.7505770
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5912561, 2.5911896
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9886684, 2.9901857
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8960142, 1.8930087
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6094437, 2.6138172
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5562820, 3.5609589
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7625809, 2.7797112
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2209215, 2.2156825

Time for backsubstitution: 13.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4121000, upper bound: 1.4171017
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4184756, upper bound: 1.4105805
time: 6.88 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3511534, 3.3514693
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7460709, 2.7436600
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5911899, 2.5912142
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9891977, 2.9886682
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8930092, 1.8940563
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6109695, 2.6094437
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5579138, 3.5562825
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7685490, 2.7625809
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2156825, 2.2175126

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170452, upper bound: 1.4117974
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4103984, upper bound: 1.4184005
time: 4.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.71 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.71
Output dim: 4, lower bound: -1.4121000, upper bound: 1.4171017
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.71
Output dim: 4, lower bound: -1.4184756, upper bound: 1.4105805
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.71
Output dim: 4, lower bound: -1.4170452, upper bound: 1.4117974
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.71
Output dim: 4, lower bound: -1.4103984, upper bound: 1.4184005

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

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4120173, upper bound: 1.4103554
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4052611, upper bound: 1.4170025
time: 6.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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

Time for backsubstitution: 12.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4183560, upper bound: 1.4037827
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4117544, upper bound: 1.4105021
time: 8.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3515663, 3.3518124
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7506285, 2.7470653
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5910826, 2.5910757
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9822168, 2.9837217
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8974996, 1.8965359
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6093998, 2.6083317
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5646019, 3.5652270
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7726240, 2.7680330
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2172108, 2.2195568

Time for backsubstitution: 12.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4105021, upper bound: 1.4117549
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170020, upper bound: 1.4052614
time: 4.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3514957, 3.3518829
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7494764, 2.7482166
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5910516, 2.5911069
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9842501, 2.9816887
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8954883, 1.8985467
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6098576, 2.6078732
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5668592, 3.5629706
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7740011, 2.7666559
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2177267, 2.2190409

Time for backsubstitution: 13.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4037810, upper bound: 1.4183563
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4103551, upper bound: 1.4120191
time: 5.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.72
Output dim: 4, lower bound: -1.4120173, upper bound: 1.4103554
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.72
Output dim: 4, lower bound: -1.4052611, upper bound: 1.4170025
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.72
Output dim: 4, lower bound: -1.4183560, upper bound: 1.4037827
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.72
Output dim: 4, lower bound: -1.4117544, upper bound: 1.4105021
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.72
Output dim: 4, lower bound: -1.4105021, upper bound: 1.4117549
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.72
Output dim: 4, lower bound: -1.4170020, upper bound: 1.4052614
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.72
Output dim: 4, lower bound: -1.4037810, upper bound: 1.4183563
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.72
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

Time for backsubstitution: 12.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1934

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4111661, upper bound: 1.4090221
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4110078, upper bound: 1.4092918
time: 4.66 seconds

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

Time for backsubstitution: 13.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1753

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 766

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3601222, upper bound: 1.3940822
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3822970, upper bound: 1.3718133
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 13.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2390

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 416

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3931922, upper bound: 1.3777511
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3931922, upper bound: 1.3777511
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 13.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 760

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4094937, upper bound: 1.4070869
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4086229, upper bound: 1.4083252
time: 5.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 13.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2378

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4018648, upper bound: 1.4020929
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4016369, upper bound: 1.4021547
time: 5.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 13.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 206

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 572

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4125101, upper bound: 1.4006845
time: 9.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4114841, upper bound: 1.4009760
time: 4.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 13.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1685

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4003022, upper bound: 1.4159255
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4017643, upper bound: 1.4151970
time: 4.46 seconds

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

Time for backsubstitution: 13.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1404

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1992

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4092407, upper bound: 1.4117591
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4102215, upper bound: 1.4104785
time: 6.18 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.4111661, upper bound: 1.4090221
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.4110078, upper bound: 1.4092918
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.3601222, upper bound: 1.3940822
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.3822970, upper bound: 1.3718133
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.3931922, upper bound: 1.3777511
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.3931922, upper bound: 1.3777511
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.4094937, upper bound: 1.4070869
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.4086229, upper bound: 1.4083252
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.4018648, upper bound: 1.4020929
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.4016369, upper bound: 1.4021547
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.4125101, upper bound: 1.4006845
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.4114841, upper bound: 1.4009760
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.4003022, upper bound: 1.4159255
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.4017643, upper bound: 1.4151970
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.4092407, upper bound: 1.4117591
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 4, lower bound: -1.4102215, upper bound: 1.4104785

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3546944, 3.3365164
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7380285, 2.7566941
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5872741, 2.5759153
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9575210, 2.9740150
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8966680, 1.8973408
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5958142, 2.5901525
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5578518, 3.5663218
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7619228, 2.7575223
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2143335, 2.2187145

Time for backsubstitution: 13.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 660

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2130

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.2902537, upper bound: 1.2869499
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.2902537, upper bound: 1.2869499
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3481874, 3.3430238
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7502337, 2.7444906
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5954266, 2.5677633
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9771838, 2.9543529
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8960133, 1.8979959
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5986981, 2.5872679
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5567465, 3.5674272
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7555799, 2.7638640
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2171574, 2.2158897

Time for backsubstitution: 13.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1257

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1934

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3967711, upper bound: 1.3903475
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3922340, upper bound: 1.3945574
time: 6.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3544035, 3.3410292
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7321157, 2.6763077
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5922818, 2.5796151
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9398212, 2.8973193
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8389215, 1.8671014
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6127472, 2.5851188
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5635948, 3.5680957
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6192350, 2.6048253
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7358603, 2.7354012
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.1979284, 2.2147410

Time for backsubstitution: 13.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 760

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1685

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3566045, upper bound: 1.3919450
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3581359, upper bound: 1.3909566
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3525591, 3.3428729
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.6675425, 2.7408834
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5990634, 2.5728335
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9045534, 2.9325848
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8617506, 1.8442709
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5945816, 2.6032841
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5630341, 3.5686569
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.5944238, 2.6296368
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7362142, 2.7350464
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2142191, 2.1984522

Time for backsubstitution: 13.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1459

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1404

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3754568, upper bound: 1.3509122
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3652517, upper bound: 1.3650848
time: 7.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3430958, 3.3469982
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7473440, 2.7536263
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5858660, 2.5870380
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9767294, 2.9839711
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8993876, 1.8909276
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5989470, 2.6104996
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5615358, 3.5689712
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7520361, 2.7768655
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2215652, 2.2147555

Time for backsubstitution: 13.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 759

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2328

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3870452, upper bound: 1.3777511
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3931921, upper bound: 1.3711861
time: 6.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3466558, 3.3479359
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7482166, 2.7527289
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5765443, 2.5910511
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9767437, 2.9852390
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8994114, 1.8920205
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5983691, 2.6127050
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5629702, 3.5670271
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7576027, 2.7795970
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2224498, 2.2134089

Time for backsubstitution: 13.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 759

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3871664, upper bound: 1.3649702
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3809188, upper bound: 1.3688910
time: 7.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3500290, 3.3565359
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7460093, 2.7325737
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5889230, 2.5969987
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9796839, 2.9828923
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.9165120, 1.9084871
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5879245, 2.5955346
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5497284, 3.5529213
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7471523, 2.7863965
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2103612, 2.2388296

Time for backsubstitution: 13.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1971

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3984919, upper bound: 1.3960874
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3976599, upper bound: 1.3962847
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3517942, 3.3550100
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7248845, 2.7537932
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5863967, 2.5994782
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9797297, 2.9828470
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.9129491, 1.9121022
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5843558, 2.5991385
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5519428, 3.5507374
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7616329, 2.7719595
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2481468, 2.2011728

Time for backsubstitution: 13.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1244

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3971076, upper bound: 1.4029239
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4025668, upper bound: 1.3980648
time: 7.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3541317, 3.3440671
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7605414, 2.7514019
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5930772, 2.5727699
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9829497, 2.9788589
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8895051, 1.8948143
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6164565, 2.6032271
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5590038, 3.5640769
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7774234, 2.7558899
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2086797, 2.2187724

Time for backsubstitution: 13.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1459

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3932039, upper bound: 1.3927526
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3932039, upper bound: 1.3927526
time: 4.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3544388, 3.3436866
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7542691, 2.7575867
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5921888, 2.5736785
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9840732, 2.9776878
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8894336, 1.8948696
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6176505, 2.6020107
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5607796, 3.5622706
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7770381, 2.7562659
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2101192, 2.2173052

Time for backsubstitution: 13.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2378

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 572

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3970943, upper bound: 1.3968499
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3960772, upper bound: 1.3971138
time: 4.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3450279, 3.3513706
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7479849, 2.7445810
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5802627, 2.5910575
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9777293, 2.9827542
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8954501, 1.8891141
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6006846, 2.6062255
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5638895, 3.5647020
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7632313, 2.7679112
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2130756, 2.2081125

Time for backsubstitution: 13.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2349

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3587210, upper bound: 1.3439542
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3587210, upper bound: 1.3439542
time: 5.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3453045, 3.3509586
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7485228, 2.7438419
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5804729, 2.5908141
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9773536, 2.9829111
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8935456, 1.8903828
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5998688, 2.6069160
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5652933, 3.5630717
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7634487, 2.7676692
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2082014, 2.2119892

Time for backsubstitution: 13.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2130

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.2891619, upper bound: 1.2802374
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.2891619, upper bound: 1.2802373
time: 5.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3566246, 3.3458204
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7470160, 2.7458000
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5870051, 2.5718341
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9851422, 2.9801779
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8883657, 1.8967056
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6026831, 2.5827761
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5631590, 3.5631895
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7799931, 2.7505283
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2114034, 2.2210927

Time for backsubstitution: 13.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2867

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 414

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3979661, upper bound: 1.4124646
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3979661, upper bound: 1.4126155
time: 4.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3561249, 3.3463202
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7463636, 2.7464511
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5911908, 2.5676484
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9894576, 2.9758627
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8873029, 1.8977685
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.5981388, 2.5873206
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5644350, 3.5619130
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7744398, 2.7560809
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2134995, 2.2189970

Time for backsubstitution: 13.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3988759, upper bound: 1.4040344
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3901752, upper bound: 1.4126911
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3368568, 3.3441939
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7194481, 2.7333493
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5811753, 2.5928471
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.9799271, 3.0056472
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8957055, 1.8952842
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6060543, 2.6090827
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5314035, 3.5189242
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7608280, 2.7599401
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2191689, 2.2183738

Time for backsubstitution: 13.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 2390

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1248

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.2154662, upper bound: 1.2197163
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.2154662, upper bound: 1.2197161
time: 6.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3379850, 3.3428323
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7349758, 2.7174625
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.5821319, 2.5918229
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -3.0039253, 2.9810438
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8956940, 1.8952887
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6037688, 2.6112256
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5242548, 3.5255547
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7582340, 2.7624414
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2203953, 2.2170501

Time for backsubstitution: 13.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1684

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2608

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3763349, upper bound: 1.3775265
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3763349, upper bound: 1.3775265
time: 4.86 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.2902537, upper bound: 1.2869499
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.2902537, upper bound: 1.2869499
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3967711, upper bound: 1.3903475
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3922340, upper bound: 1.3945574
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3566045, upper bound: 1.3919450
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3581359, upper bound: 1.3909566
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3754568, upper bound: 1.3509122
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3652517, upper bound: 1.3650848
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3870452, upper bound: 1.3777511
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3931921, upper bound: 1.3711861
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3871664, upper bound: 1.3649702
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3809188, upper bound: 1.3688910
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3984919, upper bound: 1.3960874
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3976599, upper bound: 1.3962847
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3971076, upper bound: 1.4029239
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.4025668, upper bound: 1.3980648
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3932039, upper bound: 1.3927526
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3932039, upper bound: 1.3927526
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3970943, upper bound: 1.3968499
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3960772, upper bound: 1.3971138
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3587210, upper bound: 1.3439542
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3587210, upper bound: 1.3439542
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.2891619, upper bound: 1.2802374
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.2891619, upper bound: 1.2802373
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3979661, upper bound: 1.4124646
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3979661, upper bound: 1.4126155
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3988759, upper bound: 1.4040344
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3901752, upper bound: 1.4126911
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.2154662, upper bound: 1.2197163
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.2154662, upper bound: 1.2197161
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3763349, upper bound: 1.3775265
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 4, lower bound: -1.3763349, upper bound: 1.3775265

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.3641653, 3.3419628
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.7409439, 2.7387815
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.6129112, 2.5774910
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -3.0011721, 2.9759374
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.8951945, 1.8969841
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.6297889, 2.5964513
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.5624990, 3.5669093
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.7772870, 2.7749484
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.2158852, 2.2331691

Time for backsubstitution: 13.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1933

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 414

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.2865075, upper bound: 1.2860625
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.2894082, upper bound: 1.2813675
time: 5.54 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.07
Output dim: 4, lower bound: -1.2865075, upper bound: 1.2860625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.07
Output dim: 4, lower bound: -1.2894082, upper bound: 1.2813675
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.2902537, upper bound: 1.2869499
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3967711, upper bound: 1.3903475
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3922340, upper bound: 1.3945574
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3566045, upper bound: 1.3919450
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3581359, upper bound: 1.3909566
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3754568, upper bound: 1.3509122
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3652517, upper bound: 1.3650848
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3870452, upper bound: 1.3777511
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3931921, upper bound: 1.3711861
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3871664, upper bound: 1.3649702
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3809188, upper bound: 1.3688910
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3984919, upper bound: 1.3960874
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3976599, upper bound: 1.3962847
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3971076, upper bound: 1.4029239
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.4025668, upper bound: 1.3980648
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3932039, upper bound: 1.3927526
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3932039, upper bound: 1.3927526
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3970943, upper bound: 1.3968499
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3960772, upper bound: 1.3971138
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3587210, upper bound: 1.3439542
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3587210, upper bound: 1.3439542
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.2891619, upper bound: 1.2802374
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.2891619, upper bound: 1.2802373
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3979661, upper bound: 1.4124646
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3979661, upper bound: 1.4126155
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3988759, upper bound: 1.4040344
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3901752, upper bound: 1.4126911
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.2154662, upper bound: 1.2197163
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.2154662, upper bound: 1.2197161
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3763349, upper bound: 1.3775265
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.07
Output dim: 4, lower bound: -1.3763349, upper bound: 1.3775265
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.8940565586090088
rel_dist={4: [-1.418532051192658, 1.418531738496264]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1483121, upper bound: 1.1597274
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597270, upper bound: 1.1483129
time: 4.46 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.38
Output dim: 4, lower bound: -1.1483121, upper bound: 1.1597274
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.38
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

Time for backsubstitution: 12.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 513

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1483104, upper bound: 1.1558982
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1444969, upper bound: 1.1597275
time: 5.86 seconds

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

Time for backsubstitution: 12.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597083, upper bound: 1.1464396
time: 6.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579157, upper bound: 1.1482931
time: 4.66 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.65 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.65
Output dim: 4, lower bound: -1.1483104, upper bound: 1.1558982
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.65
Output dim: 4, lower bound: -1.1444969, upper bound: 1.1597275
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.65
Output dim: 4, lower bound: -1.1597083, upper bound: 1.1464396
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.65
Output dim: 4, lower bound: -1.1579157, upper bound: 1.1482931

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0864329, 3.0802836
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4974060, 2.4971457
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3785820, 2.3674707
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7289605, 2.7262833
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7201591, 1.7226357
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3627877, 2.3554049
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2686110, 3.2714095
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4564071, 2.4614487
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4936924, 2.4850128
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0578284, 2.0617120

Time for backsubstitution: 12.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1482919, upper bound: 1.1540897
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1464379, upper bound: 1.1558793
time: 6.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0863929, 3.0803237
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4967489, 2.4978034
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3785639, 2.3674886
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7301221, 2.7251217
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7190104, 1.7237849
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3630490, 2.3551431
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2699003, 3.2701197
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4564476, 2.4614077
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4944792, 2.4842260
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0581236, 2.0614169

Time for backsubstitution: 12.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1444784, upper bound: 1.1579143
time: 6.82 seconds

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

Time for backsubstitution: 12.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

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
time: 5.38 seconds

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

Time for backsubstitution: 12.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579135, upper bound: 1.1444779
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1540905, upper bound: 1.1482921
time: 6.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.05
Output dim: 4, lower bound: -1.1482919, upper bound: 1.1540897
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.05
Output dim: 4, lower bound: -1.1464379, upper bound: 1.1558793
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.05
Output dim: 4, lower bound: -1.1444784, upper bound: 1.1579143
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.05
Output dim: 4, lower bound: -1.1426245, upper bound: 1.1597061
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.05
Output dim: 4, lower bound: -1.1597061, upper bound: 1.1426269
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.05
Output dim: 4, lower bound: -1.1558791, upper bound: 1.1464404
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.05
Output dim: 4, lower bound: -1.1579135, upper bound: 1.1444779
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.05
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

Time for backsubstitution: 12.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1244

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1459240, upper bound: 1.1467435
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1409385, upper bound: 1.1517280
time: 5.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 12.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1753

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2852

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1443445, upper bound: 1.1557498
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1462782, upper bound: 1.1538131
time: 6.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 12.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 2136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2237

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1401393, upper bound: 1.1529773
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1401912, upper bound: 1.1529307
time: 5.38 seconds

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

Time for backsubstitution: 12.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2608

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1274802, upper bound: 1.1445680
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1274802, upper bound: 1.1445680
time: 5.32 seconds

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

Time for backsubstitution: 12.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1244

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1933

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1576244, upper bound: 1.1370972
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1563804, upper bound: 1.1405852
time: 5.98 seconds

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

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1684

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1482540, upper bound: 1.1416133
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1507201, upper bound: 1.1379173
time: 5.31 seconds

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

Time for backsubstitution: 12.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1684

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 759

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1500067, upper bound: 1.1438462
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1572611, upper bound: 1.1364789
time: 5.43 seconds

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

Time for backsubstitution: 13.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1244

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1839

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1528008, upper bound: 1.1473513
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1530214, upper bound: 1.1469118
time: 6.08 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1459240, upper bound: 1.1467435
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1409385, upper bound: 1.1517280
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1443445, upper bound: 1.1557498
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1462782, upper bound: 1.1538131
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1401393, upper bound: 1.1529773
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1401912, upper bound: 1.1529307
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1274802, upper bound: 1.1445680
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1274802, upper bound: 1.1445680
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1576244, upper bound: 1.1370972
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1563804, upper bound: 1.1405852
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1482540, upper bound: 1.1416133
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1507201, upper bound: 1.1379173
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1500067, upper bound: 1.1438462
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1572611, upper bound: 1.1364789
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1528008, upper bound: 1.1473513
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.19
Output dim: 4, lower bound: -1.1530214, upper bound: 1.1469118

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0854988, 3.0774801
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4768233, 2.4802647
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3903441, 2.3816953
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7296677, 2.7278969
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7000434, 1.6995797
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3497396, 2.3447909
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2675743, 3.2731385
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4469290, 2.4515777
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4730129, 2.4767182
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0482788, 2.0514948

Time for backsubstitution: 13.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1395

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 660

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1405811, upper bound: 1.1465373
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457224, upper bound: 1.1414783
time: 4.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0841484, 3.0788293
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4765725, 2.4805157
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3928447, 2.3791955
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7297077, 2.7278571
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6988208, 1.7008018
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3496747, 2.3448558
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2676668, 3.2730470
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4455519, 2.4529545
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4756088, 2.4741216
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0506058, 2.0491681

Time for backsubstitution: 12.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1789

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1402875, upper bound: 1.1507983
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1400106, upper bound: 1.1511268
time: 5.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0857239, 3.0795510
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4965563, 2.4939542
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3780332, 2.3670604
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7255726, 2.7226548
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7173643, 1.7210310
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3596306, 2.3512177
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2651749, 3.2670417
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4558563, 2.4599688
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4935942, 2.4789507
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0574131, 2.0628119

Time for backsubstitution: 12.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1839

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 766

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1183761, upper bound: 1.1436486
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1323433, upper bound: 1.1296862
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0853844, 3.0798907
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4966259, 2.4938850
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3781457, 2.3669479
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7258606, 2.7223666
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7175074, 1.7208879
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3601265, 2.3507223
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2658730, 3.2663436
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4555125, 2.4603126
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4935989, 2.4789462
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0570979, 2.0631270

Time for backsubstitution: 12.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 765

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2333

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1306617, upper bound: 1.1444200
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1375749, upper bound: 1.1377671
time: 5.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0855274, 3.0782201
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4910765, 2.4977310
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3742905, 2.3648920
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7253823, 2.7205839
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7189999, 1.7225525
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3598437, 2.3536100
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2680035, 3.2708611
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4540429, 2.4608040
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4868231, 2.4860344
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0563531, 2.0575840

Time for backsubstitution: 12.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1404

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1395

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1292862, upper bound: 1.1414437
time: 25.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1269966, upper bound: 1.1417500
time: 9.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0848665, 3.0789387
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4930086, 2.4960847
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3760052, 2.3633535
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7248521, 2.7212486
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7195683, 1.7220571
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3591228, 2.3544369
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2679682, 3.2709050
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4550395, 2.4599872
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4865713, 2.4863584
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0572848, 2.0568044

Time for backsubstitution: 12.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 424

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1753

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1398520, upper bound: 1.1525495
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1398562, upper bound: 1.1525828
time: 7.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0861883, 3.0803056
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4960065, 2.4953458
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3785005, 2.3668885
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7301030, 2.7256870
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7177510, 1.7236872
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3629460, 2.3551941
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2697678, 3.2704248
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4560008, 2.4607596
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4940720, 2.4754992
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0561576, 2.0628996

Time for backsubstitution: 12.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1971

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1257

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1060042, upper bound: 1.1200506
time: 8.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1060042, upper bound: 1.1200506
time: 8.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0860586, 3.0803237
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4967027, 2.4953921
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3785391, 2.3674500
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7301221, 2.7245736
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7178655, 1.7237849
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3630490, 2.3535128
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2699003, 3.2683573
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4563847, 2.4608226
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4944792, 2.4778497
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0562935, 2.0612803

Time for backsubstitution: 13.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2922

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1839

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1261424, upper bound: 1.1432356
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1262604, upper bound: 1.1427527
time: 6.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0707893, 3.0776179
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4789448, 2.4817126
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3703389, 2.3822749
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7117343, 2.7174561
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7154458, 1.7091284
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3518219, 2.3617582
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2671108, 3.2697153
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4430661, 2.4396036
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4842067, 2.5019391
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0563078, 2.0492883

Time for backsubstitution: 13.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 206

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2390

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1476810, upper bound: 1.1302405
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1476810, upper bound: 1.1305355
time: 5.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0724049, 3.0760496
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4790154, 2.4818599
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3718052, 2.3808744
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7118983, 2.7174411
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7154620, 1.7091656
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3517447, 2.3618813
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2672195, 3.2696304
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4435625, 2.4393220
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4823098, 2.5040302
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0554199, 2.0502982

Time for backsubstitution: 13.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2390

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1469221, upper bound: 1.1333980
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1469221, upper bound: 1.1336334
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0420628, 3.0573812
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4471822, 2.4500248
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3477960, 2.3772707
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7090330, 2.7061222
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7140875, 1.7164402
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3313785, 2.3323412
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1802435, 3.1828084
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4565086, 2.4467809
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4748683, 2.4785707
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0522642, 2.0462134

Time for backsubstitution: 13.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 660

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2390

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1392032, upper bound: 1.1345976
time: 9.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1392032, upper bound: 1.1348970
time: 7.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0515289, 3.0474019
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4457936, 2.4508061
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3660975, 2.3583674
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7027397, 2.7120945
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7201505, 1.7100511
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3228049, 2.3404970
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1814728, 3.1801839
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4506073, 2.4522438
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4605145, 2.4923785
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0527740, 2.0455425

Time for backsubstitution: 13.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2528

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1846

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1482801, upper bound: 1.1333369
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1463703, upper bound: 1.1353227
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0949445, 3.0983422
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.3483901, 2.3048942
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3589754, 2.3695445
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7649307, 2.7573452
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7406571, 1.7389693
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2843523, 2.3104377
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2709513, 3.2689681
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4796405, 2.4680843
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4637775, 2.4570675
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0558200, 2.0580430

Time for backsubstitution: 13.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1257

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2528

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1446309, upper bound: 1.1399180
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1458458, upper bound: 1.1397850
time: 7.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0922456, 3.1010413
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.3083415, 2.3449426
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3589463, 2.3695738
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7530355, 2.7692401
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7425387, 1.7370877
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3044033, 2.2903872
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2707520, 3.2691669
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4733982, 2.4743268
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4531965, 2.4676490
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0593510, 2.0545120

Time for backsubstitution: 13.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1244

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2528

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1519349, upper bound: 1.1325520
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1531465, upper bound: 1.1324155
time: 5.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0966511, 3.1035075
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4903536, 2.4921267
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3626685, 2.3728490
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.7139845, 2.7180204
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.7221389, 1.7219472
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.3525791, 2.3582540
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.2702789, 3.2625003
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.4614048, 2.4559588
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4899564, 2.4924378
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0705755, 2.0680723

Time for backsubstitution: 13.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 3105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 894

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1450766, upper bound: 1.1399952
time: 10.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1450766, upper bound: 1.1399951
time: 5.27 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 29.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1405811, upper bound: 1.1465373
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1457224, upper bound: 1.1414783
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1402875, upper bound: 1.1507983
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1400106, upper bound: 1.1511268
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1183761, upper bound: 1.1436486
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1323433, upper bound: 1.1296862
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1306617, upper bound: 1.1444200
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1375749, upper bound: 1.1377671
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1292862, upper bound: 1.1414437
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1269966, upper bound: 1.1417500
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1398520, upper bound: 1.1525495
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1398562, upper bound: 1.1525828
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1060042, upper bound: 1.1200506
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1060042, upper bound: 1.1200506
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1261424, upper bound: 1.1432356
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1262604, upper bound: 1.1427527
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1476810, upper bound: 1.1302405
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1476810, upper bound: 1.1305355
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1469221, upper bound: 1.1333980
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1469221, upper bound: 1.1336334
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1392032, upper bound: 1.1345976
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1392032, upper bound: 1.1348970
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1482801, upper bound: 1.1333369
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1463703, upper bound: 1.1353227
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1446309, upper bound: 1.1399180
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1458458, upper bound: 1.1397850
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1519349, upper bound: 1.1325520
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1531465, upper bound: 1.1324155
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1450766, upper bound: 1.1399952
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.36
Output dim: 4, lower bound: -1.1450766, upper bound: 1.1399951
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.36
Output dim: 4, lower bound: -1.1530214, upper bound: 1.1469118
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.7199985980987549
rel_dist={4: [-1.1597516071645249, 1.1597543492710418]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516594, upper bound: 1.0487670
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0487664, upper bound: 1.0516592
time: 7.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.45 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.45
Output dim: 4, lower bound: -1.0516594, upper bound: 1.0487670
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.45
Output dim: 4, lower bound: -1.0487664, upper bound: 1.0516592

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -2.9975662, 2.9975357
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4135046, 2.4130106
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3070707, 2.3070571
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6443272, 2.6451981
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6653204, 1.6644585
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2801218, 2.2803187
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1718702, 3.1728373
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3868246, 2.3868558
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4001646, 2.4007547
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0086684, 2.0088897

Time for backsubstitution: 13.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516419, upper bound: 1.0468529
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0497452, upper bound: 1.0487497
time: 6.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -2.9975357, 2.9975657
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4130106, 2.4135041
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3070574, 2.3070705
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6451979, 2.6443269
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6644588, 1.6653204
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2803192, 2.2801223
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1728373, 3.1718702
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3868556, 2.3868251
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.4007540, 2.4001646
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0088897, 2.0086684

Time for backsubstitution: 13.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 513

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0393127, upper bound: 1.0516535
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0487596, upper bound: 1.0422051
time: 6.67 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.43
Output dim: 4, lower bound: -1.0516419, upper bound: 1.0468529
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 25.43
Output dim: 4, lower bound: -1.0497452, upper bound: 1.0487497
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.43
Output dim: 4, lower bound: -1.0393127, upper bound: 1.0516535
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 25.43
Output dim: 4, lower bound: -1.0487596, upper bound: 1.0422051

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -2.9976387, 2.9972191
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4110928, 2.4135637
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3070750, 2.3070326
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6437988, 2.6453207
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6655612, 1.6634111
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2785954, 2.2806664
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1702385, 3.1732106
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3862400, 2.3870091
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.3941965, 2.4021280
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0090837, 2.0070596

Time for backsubstitution: 13.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0421868, upper bound: 1.0468458
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516351, upper bound: 1.0373963
time: 7.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -2.9962964, 2.9917445
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4126320, 2.4134231
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3047862, 2.2964795
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6444001, 2.6406493
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6609912, 1.6645718
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2787538, 2.2728238
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1713953, 3.1715598
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3820858, 2.3858058
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.3988008, 2.3911111
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0054564, 2.0079267

Time for backsubstitution: 13.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0392952, upper bound: 1.0497412
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0373954, upper bound: 1.0516380
time: 5.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 24.46
Output dim: 4, lower bound: -1.0421868, upper bound: 1.0468458
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.46
Output dim: 4, lower bound: -1.0516351, upper bound: 1.0373963
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 24.46
Output dim: 4, lower bound: -1.0392952, upper bound: 1.0497412
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.46
Output dim: 4, lower bound: -1.0373954, upper bound: 1.0516380

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 13.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1778

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516351, upper bound: 1.0351385
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0493798, upper bound: 1.0373951
time: 6.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 1244

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2852

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0357299, upper bound: 1.0514299
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0371876, upper bound: 1.0499744
time: 5.81 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.47 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.47
Output dim: 4, lower bound: -1.0516351, upper bound: 1.0351385
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.47
Output dim: 4, lower bound: -1.0493798, upper bound: 1.0373951
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.47
Output dim: 4, lower bound: -1.0357299, upper bound: 1.0514299
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.47
Output dim: 4, lower bound: -1.0371876, upper bound: 1.0499744

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -2.9918180, 2.9959800
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4110107, 2.4131851
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.2964845, 2.3047631
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6401210, 2.6445222
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6648118, 1.6599431
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2712970, 2.2791014
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1699286, 3.1717672
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3852210, 2.3822391
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.3851418, 2.4001734
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0083413, 2.0036261

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 759

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0455716, upper bound: 1.0346700
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0511552, upper bound: 1.0290814
time: 4.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -2.9955025, 2.9910121
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.4117813, 2.4102144
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.3042374, 2.2960410
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6410122, 2.6369486
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6581960, 1.6629314
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2757206, 2.2686362
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1681347, 3.1671920
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3815351, 2.3844118
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.3987026, 2.3850479
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0049629, 2.0090265

Time for backsubstitution: 12.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 759
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1459

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1846

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0333853, upper bound: 1.0474820
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0317959, upper bound: 1.0491006
time: 5.41 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.50 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.50
Output dim: 4, lower bound: -1.0455716, upper bound: 1.0346700
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.50
Output dim: 4, lower bound: -1.0511552, upper bound: 1.0290814
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.50
Output dim: 4, lower bound: -1.0333853, upper bound: 1.0474820
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.50
Output dim: 4, lower bound: -1.0317959, upper bound: 1.0491006

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.0047307, 3.0109174
1: -7.3978786, -4.1556597, -7.3978786, -4.1556597, -2.2215695, 2.2537785
2: -7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.2879653, 2.2962654
3: -11.2633400, -7.7441711, -11.2633400, -7.7441711, -2.6710072, 2.6843305
4: 6.5621042, 8.8026104, 6.5621042, 8.8026104, -1.6847718, 1.6784914
5: -8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.2155447, 2.2083111
6: -12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.1706772, 3.1726661
7: -3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.3974419, 2.3991423
8: -6.9675961, -3.5078919, -6.9675961, -3.5078919, -2.3541126, 2.3770800
9: -5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.0073791, 2.0000153

Time for backsubstitution: 12.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 1685
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 3105
type: RSZ, layer: 3, pos: 1802
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1943
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1404
type: RSZ, layer: 3, pos: 1992
type: RSZ, layer: 3, pos: 416
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 3112
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1257
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1753
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 572
type: RSZ, layer: 3, pos: 2378
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1153
type: RSZ, layer: 3, pos: 1778
type: RSZ, layer: 3, pos: 2922
type: RSZ, layer: 3, pos: 176
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2244
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2136
type: RSZ, layer: 3, pos: 765
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1933
type: RSZ, layer: 3, pos: 1982
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 397
type: RSZ, layer: 3, pos: 2237
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2608
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1684
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2390
type: RSZ, layer: 3, pos: 1253
type: RSZ, layer: 3, pos: 166
type: RSZ, layer: 3, pos: 1449
type: RSZ, layer: 3, pos: 766
type: RSZ, layer: 3, pos: 1395
type: RSZ, layer: 3, pos: 1247
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 1459

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0494093, upper bound: 1.0236443
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0457025, upper bound: 1.0273457
time: 6.07 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 23.76 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.76
Output dim: 4, lower bound: -1.0494093, upper bound: 1.0236443
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.76
Output dim: 4, lower bound: -1.0457025, upper bound: 1.0273457
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.6619789600372314
rel_dist={4: [-1.051660938425652, 1.051661396029865]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1901.33 seconds
