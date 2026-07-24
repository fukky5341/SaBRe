## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.4091261895
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.4819937, -9.2464142, -11.4819937, -9.2464142, -2.1092680, 2.1092680)
1: (-6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.8108959, 1.8108959)
2: (-6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.9737930, 1.9737935)
3: (-5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.5463741, 1.5463738)
4: (-7.4061127, -5.1482797, -7.4061127, -5.1482797, -2.0375955, 2.0375955)
5: (-10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.7322063, 1.7322066)
6: (-17.1402016, -14.7059708, -17.1402016, -14.7059708, -2.1608863, 2.1608863)
7: (5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.2113075, 1.2113075)
8: (-6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.7075801, 1.7075803)
9: (-5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.6667328, 1.6667328)

## BASE Result
execution time: IAR + LP analysis = 14.48 + 35.38 = 49.86 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3550.14 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.0533958673477173
rel_dist={7: [-0.5945849729471151, 0.5945848479559466]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=0.9420795440673828
rel_dist={7: [-0.411179151475503, 0.41117864435442364]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=0.8678686618804932
rel_dist={7: [-0.27435192406765996, 0.27435410824511663]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=0.9049742221832275
rel_dist={7: [-0.34411985992691374, 0.34412230382054965]}

## Binary Search Result
Binary search time: 200.07 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3350.07 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526839, upper bound: 0.6517507
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6517482, upper bound: 0.6526839
time: 5.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.09 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.09
Output dim: 7, lower bound: -0.6526839, upper bound: 0.6517507
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.09
Output dim: 7, lower bound: -0.6517482, upper bound: 0.6526839

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6274064, 1.6320407
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6444960, 1.6431620
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6228499, 1.6309898
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2409601, 1.2405664
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5840011, 1.5966289
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3499680, 1.3409255
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6514335, 1.6650350
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0899401, 1.0868917
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3523965, 1.3445902
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4715693, 1.4640346

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6510476, upper bound: 0.6517461
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526798, upper bound: 0.6500683
time: 3.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6320412, 1.6274064
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6431623, 1.6444960
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6309900, 1.6228497
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2405665, 1.2409601
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5966287, 1.5840013
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3409252, 1.3499682
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6650348, 1.6514331
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0868917, 1.0899400
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3445902, 1.3523965
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4640346, 1.4715695

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6500657, upper bound: 0.6526821
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6517436, upper bound: 0.6510499
time: 4.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 20.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.57
Output dim: 7, lower bound: -0.6510476, upper bound: 0.6517461
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.57
Output dim: 7, lower bound: -0.6526798, upper bound: 0.6500683
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.57
Output dim: 7, lower bound: -0.6500657, upper bound: 0.6526821
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.57
Output dim: 7, lower bound: -0.6517436, upper bound: 0.6510499

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6286767, 1.6338956
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6330409, 1.6192267
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5976582, 1.6166134
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2245100, 1.2288938
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5641904, 1.5676959
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3362021, 1.3197324
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6491256, 1.6713481
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0917544, 1.0879276
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3512156, 1.3437526
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4268053, 1.3962812

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6510409, upper bound: 0.6445740
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6428964, upper bound: 0.6517367
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6292622, 1.6333110
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6205602, 1.6317198
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6084976, 1.6057978
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2292876, 1.2241162
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5550680, 1.5768218
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3287749, 1.3271787
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6577649, 1.6627271
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0909760, 1.0887064
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3515599, 1.3434091
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4038157, 1.4193144

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526730, upper bound: 0.6428942
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455074, upper bound: 0.6500614
time: 4.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6333106, 1.6292620
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6317201, 1.6205606
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6057978, 1.6084974
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2241163, 1.2292876
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5768218, 1.5550684
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3271785, 1.3287752
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6627274, 1.6577644
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0887065, 1.0909760
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3434088, 1.3515598
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4193144, 1.4038157

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6428964, upper bound: 0.6455099
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6428938, upper bound: 0.6526755
time: 3.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6338952, 1.6286767
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6192269, 1.6330411
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6166134, 1.5976577
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2288938, 1.2245100
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5676956, 1.5641906
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3197322, 1.3362024
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6713481, 1.6491252
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0879276, 1.0917544
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3437521, 1.3512155
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.3962812, 1.4268053

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445742, upper bound: 0.6438775
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445716, upper bound: 0.6510433
time: 3.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 7, lower bound: -0.6510409, upper bound: 0.6445740
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 7, lower bound: -0.6428964, upper bound: 0.6517367
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 7, lower bound: -0.6526730, upper bound: 0.6428942
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 7, lower bound: -0.6455074, upper bound: 0.6500614
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 7, lower bound: -0.6428964, upper bound: 0.6455099
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 7, lower bound: -0.6428938, upper bound: 0.6526755
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 7, lower bound: -0.6445742, upper bound: 0.6438775
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 7, lower bound: -0.6445716, upper bound: 0.6510433

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6263134, 1.6344461
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6300788, 1.6199279
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5957727, 1.6170547
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2246277, 1.2284166
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5623870, 1.5681262
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3341351, 1.3202128
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6470189, 1.6718357
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0923972, 1.0851915
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3479011, 1.3445275
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4277446, 1.3923130

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6510352, upper bound: 0.6419591
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6485365, upper bound: 0.6445678
time: 3.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6286767, 1.6315320
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6330409, 1.6162643
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5976582, 1.6147282
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2240326, 1.2288938
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5641904, 1.5658925
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3362021, 1.3176651
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6491256, 1.6692419
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0890183, 1.0879276
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3512156, 1.3404381
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4228370, 1.3962812

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6438696, upper bound: 0.6491243
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6413711, upper bound: 0.6517330
time: 3.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6268981, 1.6338615
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6175981, 1.6324205
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6066122, 1.6062391
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2294056, 1.2236390
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5532651, 1.5772521
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3267078, 1.3276591
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6556582, 1.6632148
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0916190, 1.0859703
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3482454, 1.3441840
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4047551, 1.4153461

Time for backsubstitution: 12.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526669, upper bound: 0.6404392
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6500744, upper bound: 0.6428908
time: 4.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6292622, 1.6309476
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6205602, 1.6287575
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6084976, 1.6039126
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2288105, 1.2241162
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5550680, 1.5750184
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3287749, 1.3251114
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6577649, 1.6606209
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0882399, 1.0887064
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3515599, 1.3400948
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.3998477, 1.4193144

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455013, upper bound: 0.6476046
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6429088, upper bound: 0.6500560
time: 4.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6309474, 1.6298125
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6287570, 1.6212616
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6039124, 1.6089385
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2242341, 1.2288104
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5750184, 1.5554986
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3251114, 1.3292556
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6606212, 1.6582522
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0893493, 1.0882399
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3400948, 1.3523347
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4202535, 1.3998475

Time for backsubstitution: 12.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6500535, upper bound: 0.6429090
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6476021, upper bound: 0.6455036
time: 3.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6333106, 1.6268985
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6317201, 1.6175983
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6057978, 1.6066120
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2236390, 1.2292876
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5768218, 1.5532650
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3271785, 1.3267078
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6627274, 1.6556582
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0859704, 1.0909760
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3434088, 1.3482454
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4153461, 1.4038157

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6428883, upper bound: 0.6500747
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6404368, upper bound: 0.6526693
time: 3.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6315320, 1.6292272
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6162639, 1.6337423
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6147280, 1.5980990
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2290118, 1.2240328
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5658927, 1.5646209
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3176651, 1.3366828
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6692419, 1.6496129
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0885706, 1.0890183
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3404381, 1.3519905
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.3972204, 1.4228370

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6517306, upper bound: 0.6413735
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6491219, upper bound: 0.6438720
time: 3.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6338952, 1.6263132
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6192269, 1.6300788
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6166134, 1.5957725
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2284167, 1.2245100
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5676956, 1.5623872
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3197322, 1.3341351
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6713481, 1.6470190
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0851917, 1.0917544
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3437521, 1.3479011
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.3923128, 1.4268053

Time for backsubstitution: 12.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445654, upper bound: 0.6485390
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6419592, upper bound: 0.6510372
time: 4.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6510352, upper bound: 0.6419591
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6485365, upper bound: 0.6445678
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6438696, upper bound: 0.6491243
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6413711, upper bound: 0.6517330
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6526669, upper bound: 0.6404392
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6500744, upper bound: 0.6428908
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6455013, upper bound: 0.6476046
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6429088, upper bound: 0.6500560
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6500535, upper bound: 0.6429090
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6476021, upper bound: 0.6455036
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6428883, upper bound: 0.6500747
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6404368, upper bound: 0.6526693
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6517306, upper bound: 0.6413735
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6491219, upper bound: 0.6438720
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6445654, upper bound: 0.6485390
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.30
Output dim: 7, lower bound: -0.6419592, upper bound: 0.6510372

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6283445, 1.6300921
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6230264, 1.6232257
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5961480, 1.6162634
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2132449, 1.2337475
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5661185, 1.5601443
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3323412, 1.3210444
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6491585, 1.6672453
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0925872, 1.0847827
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3438957, 1.3464084
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4308000, 1.3858109

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6510345, upper bound: 0.6414489
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6505641, upper bound: 0.6419574
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6219592, 1.6344461
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6300788, 1.6128759
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5949807, 1.6170547
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2246277, 1.2170336
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5544055, 1.5681262
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3341351, 1.3184190
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6424289, 1.6718357
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0919886, 1.0851915
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3479011, 1.3405221
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4212422, 1.3923130

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6485359, upper bound: 0.6440576
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6480530, upper bound: 0.6445671
time: 4.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6307082, 1.6271780
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6259894, 1.6195624
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5980334, 1.6139367
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2126498, 1.2342248
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5679214, 1.5579106
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3344088, 1.3184967
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6512647, 1.6646514
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0892084, 1.0875187
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3472102, 1.3423190
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4258924, 1.3897791

Time for backsubstitution: 12.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6438689, upper bound: 0.6486141
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6433988, upper bound: 0.6491236
time: 3.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6243229, 1.6315320
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6330409, 1.6092124
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5968661, 1.6147282
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2240326, 1.2175108
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5562088, 1.5658925
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3362021, 1.3158712
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6445351, 1.6692419
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0886097, 1.0879276
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3512156, 1.3364327
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4163346, 1.3962812

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6413705, upper bound: 0.6512227
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6408876, upper bound: 0.6517323
time: 4.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6289296, 1.6295075
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6105466, 1.6357186
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6069884, 1.6054478
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2180226, 1.2289701
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5569966, 1.5692703
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3249140, 1.3284907
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6577978, 1.6586244
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0918088, 1.0855615
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3442400, 1.3460650
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4078107, 1.4088440

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526662, upper bound: 0.6399268
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6521839, upper bound: 0.6404387
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6225443, 1.6338615
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6175981, 1.6253686
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6058211, 1.6062391
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2294056, 1.2122561
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5452836, 1.5772521
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3267078, 1.3258653
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6510677, 1.6632148
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0912101, 1.0859703
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3482454, 1.3401785
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.3982530, 1.4153461

Time for backsubstitution: 12.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6500738, upper bound: 0.6423730
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6495862, upper bound: 0.6428902
time: 3.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6312938, 1.6265936
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6135087, 1.6320553
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6088738, 1.6031213
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2174275, 1.2294471
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5587995, 1.5670366
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3269811, 1.3259432
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6599040, 1.6560305
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0884299, 1.0882976
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3475544, 1.3419757
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4029031, 1.4128122

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455006, upper bound: 0.6470919
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6450185, upper bound: 0.6476039
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6249080, 1.6309476
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6205602, 1.6217055
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6077065, 1.6039126
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2288105, 1.2127333
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5470865, 1.5750184
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3287749, 1.3233175
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6531744, 1.6606209
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0878313, 1.0887064
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3515599, 1.3360893
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.3933454, 1.4193144

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6429082, upper bound: 0.6495383
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6424209, upper bound: 0.6500553
time: 4.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6329789, 1.6254585
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6217055, 1.6245594
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6042886, 1.6081474
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2128513, 1.2341413
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5787499, 1.5475168
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3233175, 1.3300872
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6627603, 1.6536617
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0895391, 1.0878311
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3360894, 1.3542156
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4233091, 1.3933454

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6500529, upper bound: 0.6424210
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6495358, upper bound: 0.6429082
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6265936, 1.6298125
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6287570, 1.6142097
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6031213, 1.6089385
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2242341, 1.2174275
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5670369, 1.5554986
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3251114, 1.3274617
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6560307, 1.6582522
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0889406, 1.0882399
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3400948, 1.3483292
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4137514, 1.3998475

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6476016, upper bound: 0.6450185
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6470894, upper bound: 0.6455029
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6353426, 1.6225445
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6246676, 1.6208961
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6061740, 1.6058207
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2122562, 1.2346187
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5805528, 1.5452831
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3253851, 1.3275394
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6648664, 1.6510680
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0861602, 1.0905671
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3394043, 1.3501263
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4184017, 1.3973136

Time for backsubstitution: 12.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6399269, upper bound: 0.6495885
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6399269, upper bound: 0.6500758
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6289573, 1.6268985
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6317201, 1.6105464
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6050067, 1.6066120
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2236390, 1.2179047
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5688398, 1.5532650
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3271785, 1.3249140
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6581368, 1.6556582
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0855618, 1.0909760
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3434088, 1.3442400
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4088438, 1.4038157

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6399269, upper bound: 0.6521862
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6399269, upper bound: 0.6526684
time: 4.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6335635, 1.6248732
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6092124, 1.6370401
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6151042, 1.5973077
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2176287, 1.2293637
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5696232, 1.5566390
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3158712, 1.3375144
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6713810, 1.6450226
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0887604, 1.0886096
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3364327, 1.3538713
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4002759, 1.4163349

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6517301, upper bound: 0.6408900
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6512202, upper bound: 0.6413709
time: 4.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6271782, 1.6292272
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6162639, 1.6266904
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6139369, 1.5980990
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2290118, 1.2126498
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5579112, 1.5646209
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3176651, 1.3348889
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6646519, 1.6496129
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0881617, 1.0890183
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3404381, 1.3479850
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.3907182, 1.4228370

Time for backsubstitution: 12.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6491213, upper bound: 0.6434011
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6486115, upper bound: 0.6438694
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6359272, 1.6219592
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6121745, 1.6333768
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6169896, 1.5949810
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2170336, 1.2298410
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5714271, 1.5544052
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3179383, 1.3349669
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6734877, 1.6424286
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0853815, 1.0913457
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3397477, 1.3497820
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.3953683, 1.4203031

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445648, upper bound: 0.6480554
time: 3.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6440550, upper bound: 0.6485383
time: 3.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6295419, 1.6263132
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6192269, 1.6230268
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6158223, 1.5957725
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2284167, 1.2131270
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5597141, 1.5623872
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3197322, 1.3323412
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6667576, 1.6470190
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0847828, 1.0917544
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3437521, 1.3438957
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.3858106, 1.4268053

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6419560, upper bound: 0.6505638
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6414463, upper bound: 0.6510368
time: 3.73 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6510345, upper bound: 0.6414489
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6505641, upper bound: 0.6419574
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6485359, upper bound: 0.6440576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6480530, upper bound: 0.6445671
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6438689, upper bound: 0.6486141
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6433988, upper bound: 0.6491236
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6413705, upper bound: 0.6512227
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6408876, upper bound: 0.6517323
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6526662, upper bound: 0.6399268
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6521839, upper bound: 0.6404387
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6500738, upper bound: 0.6423730
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6495862, upper bound: 0.6428902
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6455006, upper bound: 0.6470919
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6450185, upper bound: 0.6476039
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6429082, upper bound: 0.6495383
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6424209, upper bound: 0.6500553
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6500529, upper bound: 0.6424210
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6495358, upper bound: 0.6429082
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6476016, upper bound: 0.6450185
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6470894, upper bound: 0.6455029
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6399269, upper bound: 0.6495885
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6399269, upper bound: 0.6500758
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6399269, upper bound: 0.6521862
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6399269, upper bound: 0.6526684
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6517301, upper bound: 0.6408900
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6512202, upper bound: 0.6413709
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6491213, upper bound: 0.6434011
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6486115, upper bound: 0.6438694
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6445648, upper bound: 0.6480554
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6440550, upper bound: 0.6485383
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6419560, upper bound: 0.6505638
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.19
Output dim: 7, lower bound: -0.6414463, upper bound: 0.6510368

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6291146, 1.6268225
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6274362, 1.6046500
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5662332, 1.6233578
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2138691, 1.2310687
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5758934, 1.5188742
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3353806, 1.3082290
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6520669, 1.6549555
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0936339, 1.0803453
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3449483, 1.3419298
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4324205, 1.3789811

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6510311, upper bound: 0.6406133
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6502628, upper bound: 0.6414459
time: 4.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6250749, 1.6300921
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6044512, 1.6232257
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5961480, 1.5863483
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2105660, 1.2337475
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5248480, 1.5601443
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3195257, 1.3210444
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6368687, 1.6672453
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0881498, 1.0847827
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3394170, 1.3464084
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4239705, 1.3858109

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6505606, upper bound: 0.6411229
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6497921, upper bound: 0.6419553
time: 3.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6227288, 1.6311760
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6344881, 1.5943000
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5650659, 1.6241488
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2252517, 1.2143549
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5641804, 1.5268555
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3371744, 1.3056035
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6453373, 1.6595461
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0930352, 1.0807540
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3489542, 1.3360434
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4228628, 1.3854835

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6485328, upper bound: 0.6432220
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6477053, upper bound: 0.6440544
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6186900, 1.6344461
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6115031, 1.6128759
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5949807, 1.5871394
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2219491, 1.2170336
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5131350, 1.5681262
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3213196, 1.3184190
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6301391, 1.6718357
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0875511, 1.0851915
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3434224, 1.3405221
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4144127, 1.3923130

Time for backsubstitution: 12.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6480499, upper bound: 0.6437316
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6472195, upper bound: 0.6445640
time: 4.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6314778, 1.6239083
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6303988, 1.6009867
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5681186, 1.6210313
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2132740, 1.2315462
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5776968, 1.5166404
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3374481, 1.3056815
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6541731, 1.6523616
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0902548, 1.0830816
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3482633, 1.3378404
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4275129, 1.3829494

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6438655, upper bound: 0.6477785
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6430973, upper bound: 0.6486098
time: 3.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6274381, 1.6271780
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6074133, 1.6195624
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5980334, 1.5840216
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2099710, 1.2342248
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5266514, 1.5579106
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3215933, 1.3184967
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6389749, 1.6646514
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0847709, 1.0875187
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3427320, 1.3423190
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4190629, 1.3897791

Time for backsubstitution: 12.71 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.090501308441162
rel_dist={7: [-0.6526852532184364, 0.6526854163544584]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741548, upper bound: 0.4735631
time: 3.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4735643, upper bound: 0.4741543
time: 7.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.04
Output dim: 7, lower bound: -0.4741548, upper bound: 0.4735631
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.04
Output dim: 7, lower bound: -0.4735643, upper bound: 0.4741543

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3386886, 1.3413368
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4402485, 1.4394865
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4177661, 1.4224174
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0568428, 1.0566177
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3260970, 1.3333130
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1299210, 1.1247537
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3533473, 1.3611196
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9773173, 0.9755752
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1301293, 1.1256685
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.3263383, 1.3220327

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732650, upper bound: 0.4735586
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741525, upper bound: 0.4726714
time: 3.80 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3413365, 1.3386886
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4394865, 1.4402487
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4224176, 1.4177659
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0566180, 1.0568428
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3333130, 1.3260972
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1247540, 1.1299212
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3611197, 1.3533472
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9755754, 0.9773172
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1256685, 1.1301293
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.3220325, 1.3263383

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726718, upper bound: 0.4741551
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4735589, upper bound: 0.4732676
time: 3.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.60
Output dim: 7, lower bound: -0.4732650, upper bound: 0.4735586
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.60
Output dim: 7, lower bound: -0.4741525, upper bound: 0.4726714
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.60
Output dim: 7, lower bound: -0.4726718, upper bound: 0.4741551
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.60
Output dim: 7, lower bound: -0.4735589, upper bound: 0.4732676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3399589, 1.3429412
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4234452, 1.4155512
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3925743, 1.4034057
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0403926, 1.0428977
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3023767, 1.3043799
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1129723, 1.1035609
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3510394, 1.3637379
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9787982, 0.9766113
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1289482, 1.1246836
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2717214, 1.2542791

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732617, upper bound: 0.4684858
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4681922, upper bound: 0.4735553
time: 5.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3402932, 1.3426071
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4163136, 1.4226902
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3987684, 1.3972254
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0431228, 1.0401676
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2971644, 1.3095948
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1087279, 1.1078157
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3559761, 1.3588117
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9783533, 0.9770563
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1291449, 1.1244874
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2585847, 1.2674410

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741491, upper bound: 0.4675986
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690796, upper bound: 0.4726684
time: 4.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3426068, 1.3402934
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4226899, 1.4163134
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3972254, 1.3987679
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0401678, 1.0431228
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3095946, 1.2971642
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1078157, 1.1087279
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3588119, 1.3559760
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9770563, 0.9783533
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1244874, 1.1291449
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2674410, 1.2585847

Time for backsubstitution: 12.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726684, upper bound: 0.4690822
time: 3.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4676016, upper bound: 0.4741516
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3429415, 1.3399589
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4155507, 1.4234452
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4034061, 1.3925738
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0428977, 1.0403926
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3043799, 1.3023769
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1035609, 1.1129723
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3637381, 1.3510393
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9766114, 0.9787980
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1246836, 1.1289482
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2542791, 1.2717214

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4684887, upper bound: 0.4681947
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4684861, upper bound: 0.4732612
time: 7.00 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.66 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.66
Output dim: 7, lower bound: -0.4732617, upper bound: 0.4684858
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.66
Output dim: 7, lower bound: -0.4681922, upper bound: 0.4735553
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.66
Output dim: 7, lower bound: -0.4741491, upper bound: 0.4675986
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.66
Output dim: 7, lower bound: -0.4690796, upper bound: 0.4726684
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.66
Output dim: 7, lower bound: -0.4726684, upper bound: 0.4690822
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.66
Output dim: 7, lower bound: -0.4676016, upper bound: 0.4741516
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.66
Output dim: 7, lower bound: -0.4684887, upper bound: 0.4681947
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.66
Output dim: 7, lower bound: -0.4684861, upper bound: 0.4732612

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3375952, 1.3422428
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4204831, 1.4146821
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3906889, 1.4028499
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0402553, 1.0424204
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3005733, 1.3038529
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1109047, 1.1029491
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3489332, 1.3631140
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9779928, 0.9738752
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1256337, 1.1237061
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2705576, 1.2503109

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732596, upper bound: 0.4665229
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4712992, upper bound: 0.4684866
time: 3.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3392603, 1.3405776
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4225764, 1.4125888
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3920183, 1.4015205
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0399153, 1.0427605
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3018498, 1.3025765
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1123605, 1.1014934
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3504152, 1.3616318
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9760621, 0.9758060
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1279705, 1.1213691
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2677534, 1.2531152

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4681902, upper bound: 0.4715921
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4662297, upper bound: 0.4735531
time: 5.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3379300, 1.3419087
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4133506, 1.4218211
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3968830, 1.3966696
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0429857, 1.0396904
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2953610, 1.3090677
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1066604, 1.1072042
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3538699, 1.3581878
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9775481, 0.9743202
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1258307, 1.1235096
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2574208, 1.2634728

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741471, upper bound: 0.4656357
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4721868, upper bound: 0.4675994
time: 3.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3395951, 1.3402436
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4154439, 1.4197278
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3982124, 1.3953400
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0426457, 1.0400305
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2966371, 1.3077914
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1081166, 1.1057484
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3553519, 1.3567055
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9756172, 0.9762510
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1281672, 1.1211731
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2546165, 1.2662771

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690776, upper bound: 0.4707081
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4671173, upper bound: 0.4726663
time: 5.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3402436, 1.3395951
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4197278, 1.4154444
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3953400, 1.3982122
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0400305, 1.0426456
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3077912, 1.2966372
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1057487, 1.1081166
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3567057, 1.3553519
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9762511, 0.9756172
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1211729, 1.1281672
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2662771, 1.2546165

Time for backsubstitution: 12.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726664, upper bound: 0.4671168
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4707054, upper bound: 0.4690801
time: 3.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3419087, 1.3379298
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4218211, 1.4133511
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3966694, 1.3968828
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0396903, 1.0429856
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3090677, 1.2953608
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1072040, 1.1066606
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3581877, 1.3538698
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9743204, 0.9775479
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1235099, 1.1258304
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2634728, 1.2574208

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4675969, upper bound: 0.4721864
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4656360, upper bound: 0.4741497
time: 3.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3405774, 1.3392606
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4125886, 1.4225762
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4015207, 1.3920181
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0427606, 1.0399153
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3025765, 1.3018498
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1014934, 1.1123605
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3616319, 1.3504152
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9758060, 0.9760619
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1213694, 1.1279705
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2531154, 1.2677531

Time for backsubstitution: 12.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4735535, upper bound: 0.4662292
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4715925, upper bound: 0.4681927
time: 3.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3422425, 1.3375955
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4146819, 1.4204829
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4028502, 1.3906887
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0424204, 1.0402554
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3038526, 1.3005735
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1029491, 1.1109047
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3631139, 1.3489331
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9738753, 0.9779928
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1237059, 1.1256337
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2503109, 1.2705576

Time for backsubstitution: 12.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4684840, upper bound: 0.4712987
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4665231, upper bound: 0.4732623
time: 3.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 20.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4732596, upper bound: 0.4665229
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4712992, upper bound: 0.4684866
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4681902, upper bound: 0.4715921
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4662297, upper bound: 0.4735531
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4741471, upper bound: 0.4656357
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4721868, upper bound: 0.4675994
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4690776, upper bound: 0.4707081
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4671173, upper bound: 0.4726663
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4726664, upper bound: 0.4671168
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4707054, upper bound: 0.4690801
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4675969, upper bound: 0.4721864
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4656360, upper bound: 0.4741497
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4735535, upper bound: 0.4662292
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4715925, upper bound: 0.4681927
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4684840, upper bound: 0.4712987
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.85
Output dim: 7, lower bound: -0.4665231, upper bound: 0.4732623

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3368902, 1.3378888
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4134307, 1.4135444
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3905644, 1.4020584
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0288725, 1.0405884
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2992847, 1.2958710
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1091113, 1.1026559
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3481884, 1.3585236
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9779263, 0.9734664
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1216283, 1.1230643
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2695169, 1.2438087

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732579, upper bound: 0.4658678
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726182, upper bound: 0.4665238
time: 3.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3332415, 1.3415376
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4193454, 1.4076302
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3898969, 1.4027255
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0384233, 1.0310376
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2925913, 1.3025641
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1106114, 1.1011553
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3443427, 1.3623692
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9775841, 0.9738085
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1249919, 1.1197004
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2640553, 1.2492704

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4712975, upper bound: 0.4678328
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4706569, upper bound: 0.4684848
time: 3.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3385553, 1.3362236
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4155240, 1.4114511
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3918939, 1.4007289
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0285325, 1.0409284
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3005612, 1.2945946
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1105666, 1.1012001
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3496704, 1.3570414
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9759953, 0.9753971
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1239650, 1.1207273
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2667127, 1.2466133

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4681884, upper bound: 0.4709395
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4675488, upper bound: 0.4715934
time: 3.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3349066, 1.3398724
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4214387, 1.4055369
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3912263, 1.4013960
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0380833, 1.0313776
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2938683, 1.3012878
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1120672, 1.0996995
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3458252, 1.3608869
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9756532, 0.9757392
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1273289, 1.1173637
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2612510, 1.2520747

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4662280, upper bound: 0.4729053
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4655874, upper bound: 0.4735543
time: 3.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3372245, 1.3375547
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4062991, 1.4206834
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3967586, 1.3958781
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0316026, 1.0378582
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2940724, 1.3010859
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1048670, 1.1069107
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3531251, 1.3535974
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9774814, 0.9739114
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1218250, 1.1228678
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2563801, 1.2569706

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741453, upper bound: 0.4649805
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4735056, upper bound: 0.4656340
time: 3.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3335757, 1.3412036
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4122128, 1.4147692
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3960910, 1.3965452
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0411534, 1.0283074
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2873790, 1.3077791
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1063671, 1.1054106
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3492794, 1.3574430
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9771392, 0.9742535
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1251888, 1.1195042
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2509184, 1.2624323

Time for backsubstitution: 12.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4721851, upper bound: 0.4669487
time: 3.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4715445, upper bound: 0.4675976
time: 3.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3388901, 1.3358896
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4083924, 1.4185901
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3980880, 1.3945487
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0312626, 1.0381984
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2953484, 1.2998095
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1063228, 1.1054549
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3546076, 1.3521152
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9755504, 0.9758422
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1241617, 1.1205313
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2535758, 1.2597752

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690758, upper bound: 0.4700532
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4684362, upper bound: 0.4707062
time: 3.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3352413, 1.3395383
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4143062, 1.4126759
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3974204, 1.3952157
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0408134, 1.0286475
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2886555, 1.3065026
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1078229, 1.1039546
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3507614, 1.3559607
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9752085, 0.9761844
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1275253, 1.1171675
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2481141, 1.2652366

Time for backsubstitution: 12.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4671156, upper bound: 0.4720183
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4664750, upper bound: 0.4726672
time: 3.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3395386, 1.3352411
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4126754, 1.4143066
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3952155, 1.3974206
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0286477, 1.0408134
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3065026, 1.2886553
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1039548, 1.1078229
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3559608, 1.3507617
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9761844, 0.9752083
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1171675, 1.1275256
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2652364, 1.2481143

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726646, upper bound: 0.4664775
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4720154, upper bound: 0.4671152
time: 6.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3358898, 1.3388898
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4185901, 1.4083924
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3945489, 1.3980877
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0381982, 1.0312626
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2998097, 1.2953484
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1054549, 1.1063228
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3521152, 1.3546071
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9758422, 0.9755504
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1205311, 1.1241617
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2597747, 1.2535758

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4707036, upper bound: 0.4684386
time: 3.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4649838, upper bound: 0.4690771
time: 4.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3412037, 1.3335758
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4147687, 1.4122133
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3965449, 1.3960912
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0283074, 1.0411534
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3077791, 1.2873789
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1054106, 1.1063671
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3574429, 1.3492794
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9742537, 0.9771391
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1195042, 1.1251886
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2624321, 1.2509189

Time for backsubstitution: 12.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4675952, upper bound: 0.4715470
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4669460, upper bound: 0.4721848
time: 3.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3375549, 1.3372246
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4206834, 1.4062991
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3958783, 1.3967583
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0378582, 1.0316026
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3010857, 1.2940720
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1069107, 1.1048670
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3535976, 1.3531249
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9739115, 0.9774811
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1228681, 1.1218250
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2569704, 1.2563803

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4656342, upper bound: 0.4735082
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4649810, upper bound: 0.4741448
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3398724, 1.3349066
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4055371, 1.4214385
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4013963, 1.3912265
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0313776, 1.0380833
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3012879, 1.2938679
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0996995, 1.1120672
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3608871, 1.3458250
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9757395, 0.9756532
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1173639, 1.1273286
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2520747, 1.2612512

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4735518, upper bound: 0.4655871
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4729026, upper bound: 0.4662275
time: 7.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3362236, 1.3385553
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4114509, 1.4155242
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4007287, 1.3918936
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0409284, 1.0285325
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2945950, 1.3005611
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1012001, 1.1105669
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3570414, 1.3496704
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9753973, 0.9759953
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1207275, 1.1239650
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2466130, 1.2667127

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4715908, upper bound: 0.4675513
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4709375, upper bound: 0.4681909
time: 4.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3415375, 1.3332415
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4076304, 1.4193451
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4027257, 1.3898971
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0310376, 1.0384233
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3025639, 1.2925916
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1011553, 1.1106114
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3623695, 1.3443427
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9738085, 0.9775840
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1197004, 1.1249919
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2492702, 1.2640555

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4684823, upper bound: 0.4706567
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4678332, upper bound: 0.4712967
time: 3.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3378887, 1.3368901
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4135442, 1.4134309
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4020581, 1.3905642
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0405884, 1.0288725
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2958710, 1.2992847
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1026559, 1.1091111
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3585238, 1.3481882
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9734664, 0.9779260
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1230640, 1.1216283
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2438087, 1.2695169

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4665213, upper bound: 0.4726207
time: 3.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4658681, upper bound: 0.4732574
time: 4.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4732579, upper bound: 0.4658678
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4726182, upper bound: 0.4665238
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4712975, upper bound: 0.4678328
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4706569, upper bound: 0.4684848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4681884, upper bound: 0.4709395
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4675488, upper bound: 0.4715934
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4662280, upper bound: 0.4729053
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4655874, upper bound: 0.4735543
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4741453, upper bound: 0.4649805
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4735056, upper bound: 0.4656340
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4721851, upper bound: 0.4669487
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4715445, upper bound: 0.4675976
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4690758, upper bound: 0.4700532
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4684362, upper bound: 0.4707062
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4671156, upper bound: 0.4720183
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4664750, upper bound: 0.4726672
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4726646, upper bound: 0.4664775
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4720154, upper bound: 0.4671152
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4707036, upper bound: 0.4684386
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4649838, upper bound: 0.4690771
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4675952, upper bound: 0.4715470
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4669460, upper bound: 0.4721848
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4656342, upper bound: 0.4735082
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4649810, upper bound: 0.4741448
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4735518, upper bound: 0.4655871
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4729026, upper bound: 0.4662275
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4715908, upper bound: 0.4675513
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4709375, upper bound: 0.4681909
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4684823, upper bound: 0.4706567
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4678332, upper bound: 0.4712967
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4665213, upper bound: 0.4726207
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.52
Output dim: 7, lower bound: -0.4658681, upper bound: 0.4732574

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3359284, 1.3346192
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4079895, 1.3949687
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3606491, 1.3932917
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0280809, 1.0379096
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2871828, 1.2546009
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1053553, 1.0898404
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3445833, 1.3462338
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9766223, 0.9690289
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1203103, 1.1185856
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2675159, 1.2369790

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732558, upper bound: 0.4653588
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4727466, upper bound: 0.4658687
time: 3.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3336205, 1.3369274
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3948550, 1.4081037
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3817973, 1.3721433
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0261936, 1.0397968
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2580147, 1.2837695
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0962954, 1.0989003
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3358986, 1.3549194
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9734888, 0.9721627
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1171498, 1.1217463
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2626874, 1.2418075

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726161, upper bound: 0.4660129
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4721070, upper bound: 0.4665217
time: 3.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3322797, 1.3382679
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4139037, 1.3890545
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3599820, 1.3939588
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0376320, 1.0283588
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2804899, 1.2612940
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1068559, 1.0883400
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3407376, 1.3500794
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9762805, 0.9693711
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1236742, 1.1152220
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2620542, 1.2424407

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4712953, upper bound: 0.4673249
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4707864, upper bound: 0.4678306
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3299718, 1.3405762
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4007692, 1.4021895
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3811307, 1.3728104
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0357447, 1.0302460
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2513218, 1.2904626
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0977960, 1.0973997
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3320529, 1.3587649
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9731467, 0.9725049
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1205134, 1.1183827
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2572258, 1.2472689

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4706547, upper bound: 0.4679737
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4701459, upper bound: 0.4684825
time: 3.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3375936, 1.3329539
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4100828, 1.3928754
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3619785, 1.3919623
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0277410, 1.0382496
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2884598, 1.2533245
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1068115, 1.0883846
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3460658, 1.3447516
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9746916, 0.9709597
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1226473, 1.1162488
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2647116, 1.2397835

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4681863, upper bound: 0.4704295
time: 3.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4676771, upper bound: 0.4709382
time: 4.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3352857, 1.3352622
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3969483, 1.4060104
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3831267, 1.3708138
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0258536, 1.0401369
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2592907, 1.2824931
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0977516, 1.0974445
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3373806, 1.3534372
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9715579, 0.9740934
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1194863, 1.1194096
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2598829, 1.2446117

Time for backsubstitution: 12.43 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=0.9791851043701172
rel_dist={7: [-0.4741558180834273, 0.47415573392575094]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111784, upper bound: 0.4107158
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107163, upper bound: 0.4111781
time: 6.03 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.84 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.84
Output dim: 7, lower bound: -0.4111784, upper bound: 0.4107158
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.84
Output dim: 7, lower bound: -0.4107163, upper bound: 0.4111781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2424493, 1.2444355
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3721662, 1.3715947
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3494048, 1.3528934
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9954703, 0.9953015
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2401290, 1.2455409
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0565722, 1.0526967
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2539849, 1.2598145
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9397764, 0.9384699
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0560400, 1.0526946
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2779279, 1.2746987

Time for backsubstitution: 12.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4105118, upper bound: 0.4107139
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111766, upper bound: 0.4100496
time: 4.47 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2444355, 1.2424494
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3715944, 1.3721664
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3528934, 1.3494046
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9953015, 0.9954703
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2455411, 1.2401291
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0526967, 1.0565722
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2598147, 1.2539852
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9384699, 0.9397763
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0526946, 1.0560400
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2746987, 1.2779279

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100503, upper bound: 0.4111763
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107171, upper bound: 0.4105146
time: 4.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.59
Output dim: 7, lower bound: -0.4105118, upper bound: 0.4107139
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.59
Output dim: 7, lower bound: -0.4111766, upper bound: 0.4100496
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.59
Output dim: 7, lower bound: -0.4100503, upper bound: 0.4111763
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.59
Output dim: 7, lower bound: -0.4107171, upper bound: 0.4105146

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2437196, 1.2459564
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3535800, 1.3476593
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3242130, 1.3323364
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9790201, 0.9808989
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2151055, 1.2166079
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0385623, 1.0315037
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2516775, 1.2612013
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9411459, 0.9395058
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0548589, 1.0516605
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2200270, 1.2069452

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4105098, upper bound: 0.4070349
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068430, upper bound: 0.4107117
time: 4.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2439704, 1.2457058
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3482308, 1.3530135
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3288584, 1.3277013
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9810677, 0.9788513
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2111959, 1.2205191
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0353789, 1.0346949
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2553802, 1.2575066
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9408123, 0.9398396
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0550065, 1.0515134
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2101743, 1.2168164

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111746, upper bound: 0.4063708
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075078, upper bound: 0.4100476
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2457058, 1.2439705
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3530135, 1.3482311
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3277016, 1.3288581
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9788513, 0.9810678
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2205191, 1.2111962
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0346949, 1.0353789
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2575068, 1.2553798
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9398396, 0.9408122
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0515134, 1.0550065
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2168164, 1.2101743

Time for backsubstitution: 12.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4063743, upper bound: 0.4075071
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4063715, upper bound: 0.4111764
time: 3.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2459564, 1.2437197
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3476596, 1.3535798
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3323364, 1.3242126
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9808989, 0.9790201
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2166076, 1.2151057
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0315037, 1.0385623
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2612014, 1.2516773
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9395058, 0.9411459
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0516605, 1.0548589
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2069452, 1.2200270

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107123, upper bound: 0.4068423
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4070356, upper bound: 0.4105092
time: 4.23 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.76
Output dim: 7, lower bound: -0.4105098, upper bound: 0.4070349
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.76
Output dim: 7, lower bound: -0.4068430, upper bound: 0.4107117
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.76
Output dim: 7, lower bound: -0.4111746, upper bound: 0.4063708
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.76
Output dim: 7, lower bound: -0.4075078, upper bound: 0.4100476
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 21.76
Output dim: 7, lower bound: -0.4063743, upper bound: 0.4075071
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.76
Output dim: 7, lower bound: -0.4063715, upper bound: 0.4111764
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.76
Output dim: 7, lower bound: -0.4107123, upper bound: 0.4068423
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.76
Output dim: 7, lower bound: -0.4070356, upper bound: 0.4105092

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2413561, 1.2448418
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3506169, 1.3462670
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3223276, 1.3314483
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9787979, 0.9804218
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2133021, 1.2157619
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0364947, 1.0305281
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2495713, 1.2602067
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9398580, 0.9367697
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0515447, 1.0500987
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2181621, 1.2029769

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4105079, upper bound: 0.4056813
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4091239, upper bound: 0.4070361
time: 3.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2426050, 1.2435929
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3521876, 1.3446970
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3233242, 1.3304513
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9785430, 0.9806768
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2142596, 1.2148045
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0375867, 1.0294361
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2506828, 1.2590951
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9384098, 0.9382179
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0532973, 1.0483463
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2160587, 1.2050803

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068411, upper bound: 0.4093452
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4054583, upper bound: 0.4107129
time: 3.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2416070, 1.2445911
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3452687, 1.3516212
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3269730, 1.3268130
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9808457, 0.9783742
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2093930, 1.2196729
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0333116, 1.0337193
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2532735, 1.2565122
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9395244, 0.9371035
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0516922, 1.0499516
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2083094, 1.2128482

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4061259, upper bound: 0.4050135
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4097888, upper bound: 0.4063720
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2428558, 1.2433423
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3468385, 1.3500512
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3279696, 1.3258159
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9805906, 0.9786292
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2103500, 1.2187157
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0344033, 1.0326276
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2543855, 1.2554004
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9380763, 0.9385517
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0534449, 1.0481989
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2062061, 1.2149515

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4075059, upper bound: 0.4086806
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4061232, upper bound: 0.4100488
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2445912, 1.2416070
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3516212, 1.3452687
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3268127, 1.3269727
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9783742, 0.9808456
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2196727, 1.2093928
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0337193, 1.0333116
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2565122, 1.2532736
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9371035, 0.9395243
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0499516, 1.0516922
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2128484, 1.2083094

Time for backsubstitution: 12.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4063694, upper bound: 0.4097882
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050141, upper bound: 0.4111727
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2435927, 1.2426050
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3446965, 1.3521876
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3304510, 1.3233244
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9806769, 0.9785429
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2148042, 1.2142596
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0294361, 1.0375867
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2590952, 1.2506827
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9382179, 0.9384098
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0483463, 1.0532973
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2050803, 1.2160587

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107102, upper bound: 0.4054574
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4093454, upper bound: 0.4068438
time: 4.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2448418, 1.2413561
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3462672, 1.3506174
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3314486, 1.3223274
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9804218, 0.9787979
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2157617, 1.2133023
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0305281, 1.0364947
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2602067, 1.2495711
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9367697, 0.9398580
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0500989, 1.0515447
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2029769, 1.2181621

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4070334, upper bound: 0.4091234
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4056786, upper bound: 0.4105076
time: 4.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4105079, upper bound: 0.4056813
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4091239, upper bound: 0.4070361
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4068411, upper bound: 0.4093452
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4054583, upper bound: 0.4107129
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4061259, upper bound: 0.4050135
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4097888, upper bound: 0.4063720
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4075059, upper bound: 0.4086806
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4061232, upper bound: 0.4100488
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4063694, upper bound: 0.4097882
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4050141, upper bound: 0.4111727
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4107102, upper bound: 0.4054574
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4093454, upper bound: 0.4068438
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4070334, upper bound: 0.4091234
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.35
Output dim: 7, lower bound: -0.4056786, upper bound: 0.4105076

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2397387, 1.2404878
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3435655, 1.3436508
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3220363, 1.3306568
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9674151, 0.9762019
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2103403, 1.2077799
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0347009, 1.0298595
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2478647, 1.2556164
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9397058, 0.9363610
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0475392, 1.0486162
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2157559, 1.1964750

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4105064, upper bound: 0.4051756
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100086, upper bound: 0.4056767
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2409875, 1.2392389
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3451352, 1.3420808
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3230329, 1.3296597
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9671600, 0.9764569
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2112973, 1.2068226
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0357928, 1.0287676
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2489767, 1.2545047
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9382577, 0.9378090
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0492916, 1.0468636
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2136526, 1.1985781

Time for backsubstitution: 12.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4068396, upper bound: 0.4088489
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4063418, upper bound: 0.4093467
time: 3.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2382510, 1.2419754
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3495708, 1.3376451
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3225331, 1.3301601
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9743230, 0.9692938
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2062781, 1.2118425
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0369182, 1.0276423
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2460923, 1.2573888
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9380012, 0.9380655
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0518146, 1.0443408
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2095566, 1.2026744

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4049619, upper bound: 0.4102087
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4045145, upper bound: 0.4107083
time: 7.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2372530, 1.2429737
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3426518, 1.3445692
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3261819, 1.3265219
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9766257, 0.9669912
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2014105, 1.2167109
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0326431, 1.0319254
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2486835, 1.2548058
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9391155, 0.9369513
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0502095, 1.0459461
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2018070, 1.2104425

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4097873, upper bound: 0.4058655
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4092896, upper bound: 0.4063707
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2385018, 1.2417248
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3442225, 1.3429992
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3271785, 1.3255248
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9763706, 0.9672463
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2023680, 1.2157536
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0337348, 1.0308337
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2497950, 1.2536942
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9376674, 0.9383993
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0519619, 1.0441935
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.1997039, 1.2125456

Time for backsubstitution: 12.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4061217, upper bound: 0.4095447
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4056240, upper bound: 0.4100441
time: 5.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2429738, 1.2372530
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3445687, 1.3426523
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3265224, 1.3261817
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9669912, 0.9766258
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2167113, 1.2014109
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0319254, 1.0326431
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2548060, 1.2486832
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9369514, 0.9391155
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0459461, 1.0502095
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2104423, 1.2018073

Time for backsubstitution: 12.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4063680, upper bound: 0.4092893
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4045145, upper bound: 0.4097872
time: 6.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2402372, 1.2399895
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3490052, 1.3382168
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3260217, 1.3266819
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9741542, 0.9694626
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2116911, 1.2064307
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0330508, 1.0315177
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2519217, 1.2515674
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9366949, 0.9393721
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0484688, 1.0476868
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2063460, 1.2059035

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4045145, upper bound: 0.4106731
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4045116, upper bound: 0.4111709
time: 5.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2419753, 1.2382510
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3376451, 1.3495712
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3301606, 1.3225329
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9692938, 0.9743230
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2118428, 1.2062776
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0276423, 1.0369182
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2573891, 1.2460923
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9380658, 0.9380010
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0443408, 1.0518143
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2026742, 1.2095566

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107087, upper bound: 0.4049618
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4051789, upper bound: 0.4054561
time: 9.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2392387, 1.2409875
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3420806, 1.3451357
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3296599, 1.3230333
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9764569, 0.9671600
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2068226, 1.2112975
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0287676, 1.0357928
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2545047, 1.2489765
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9378092, 0.9382576
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0468636, 1.0492918
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.1985779, 1.2136528

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4093440, upper bound: 0.4063412
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4088462, upper bound: 0.4068390
time: 4.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2404878, 1.2397387
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3436503, 1.3435655
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3306565, 1.3220360
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9762018, 0.9674151
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2077801, 1.2103403
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0298595, 1.0347009
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2556162, 1.2478648
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9363611, 0.9397057
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0486159, 1.0475390
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.1964748, 1.2157559

Time for backsubstitution: 12.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4056772, upper bound: 0.4100083
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4051761, upper bound: 0.4105061
time: 6.93 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4105064, upper bound: 0.4051756
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4100086, upper bound: 0.4056767
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4068396, upper bound: 0.4088489
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4063418, upper bound: 0.4093467
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4049619, upper bound: 0.4102087
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4045145, upper bound: 0.4107083
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4097873, upper bound: 0.4058655
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4092896, upper bound: 0.4063707
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4061217, upper bound: 0.4095447
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4056240, upper bound: 0.4100441
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4063680, upper bound: 0.4092893
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4045145, upper bound: 0.4097872
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4045145, upper bound: 0.4106731
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4045116, upper bound: 0.4111709
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4107087, upper bound: 0.4049618
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4051789, upper bound: 0.4054561
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4093440, upper bound: 0.4063412
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4088462, upper bound: 0.4068390
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4056772, upper bound: 0.4100083
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 7, lower bound: -0.4051761, upper bound: 0.4105061

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2382002, 1.2372181
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3348403, 1.3250751
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.2921209, 1.3166029
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9661517, 0.9735231
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1909466, 1.1665099
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0286803, 1.0170441
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2420886, 1.2433267
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9376187, 0.9319235
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0454311, 1.0441375
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2125478, 1.1896453

Time for backsubstitution: 12.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4105045, upper bound: 0.4047911
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4101219, upper bound: 0.4051739
time: 4.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2364690, 1.2389493
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3249898, 1.3349261
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3079824, 1.3007417
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9647365, 0.9749386
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1690700, 1.1883863
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0218854, 1.0238390
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2355750, 1.2498407
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9352684, 0.9342738
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0430605, 1.0465081
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2089264, 1.1932664

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100067, upper bound: 0.4052927
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4096241, upper bound: 0.4056752
time: 4.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2377179, 1.2377003
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3265595, 1.3333561
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3089795, 1.2997446
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9644814, 0.9751936
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1700275, 1.1874290
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0229774, 1.0227470
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2366869, 1.2487291
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9338202, 0.9357219
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0448132, 1.0447555
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2068231, 1.1953697

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4045774, upper bound: 0.4089621
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4059573, upper bound: 0.4093416
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2367125, 1.2387058
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3408461, 1.3190694
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.2926178, 1.3161063
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9730597, 0.9666151
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1868839, 1.1705723
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0308976, 1.0148270
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2403162, 1.2450991
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9359140, 0.9336281
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0497065, 1.0398622
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2063482, 1.1958447

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4054549, upper bound: 0.4098247
time: 6.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050723, upper bound: 0.4102067
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2349813, 1.2404369
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3309956, 1.3289206
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3084793, 1.3002450
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9716444, 0.9680306
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1650076, 1.1924489
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0241027, 1.0216219
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2338026, 1.2516133
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9335637, 0.9359784
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0473359, 1.0422328
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2027268, 1.1994658

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4049572, upper bound: 0.4103271
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4045746, upper bound: 0.4107095
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2357144, 1.2397040
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3339276, 1.3259935
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.2962666, 1.3124685
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9753623, 0.9643124
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1820171, 1.1754408
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0266225, 1.0191102
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2429078, 1.2425160
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9370284, 0.9325138
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0481014, 1.0414674
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.1985991, 1.2036128

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4097854, upper bound: 0.4054811
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4052423, upper bound: 0.4058668
time: 4.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2339833, 1.2414353
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3240767, 1.3358443
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3121271, 1.2966068
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9739470, 0.9657279
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1601408, 1.1973172
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0198276, 1.0259051
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2363937, 1.2490299
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9346781, 0.9348642
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0457308, 1.0438380
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.1949775, 1.2072339

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4092877, upper bound: 0.4059863
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4052423, upper bound: 0.4063658
time: 4.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2369635, 1.2384552
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3354979, 1.3244236
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.2972636, 1.3114715
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9751077, 0.9645675
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1829746, 1.1744835
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0277145, 1.0180182
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2440193, 1.2414044
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9355803, 0.9339619
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0498540, 1.0397148
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.1964958, 1.2057159

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4052423, upper bound: 0.4091635
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057372, upper bound: 0.4095430
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2352321, 1.2401863
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3256464, 1.3342741
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3131242, 1.2956097
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9736919, 0.9659830
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1610980, 1.1963599
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0209196, 1.0248132
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2375052, 1.2479182
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9332299, 0.9363122
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0474834, 1.0420856
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.1928742, 1.2093372

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4056221, upper bound: 0.4096616
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4052395, upper bound: 0.4100454
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2414353, 1.2339833
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3358440, 1.3240767
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.2966065, 1.3121274
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9657278, 0.9739470
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1973171, 1.1601408
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0259051, 1.0198276
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2490299, 1.2363935
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9348643, 0.9346781
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0438380, 1.0457308
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2072337, 1.1949775

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4063660, upper bound: 0.4089049
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4059835, upper bound: 0.4092874
time: 4.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2397039, 1.2357144
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3259935, 1.3339279
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3124685, 1.2962666
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9643126, 0.9753624
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1754408, 1.1820172
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0191102, 1.0266225
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2425163, 1.2429075
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9325140, 0.9370284
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0414677, 1.0481014
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2036126, 1.1985991

Time for backsubstitution: 12.63 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=0.9420795440673828
rel_dist={7: [-0.411179151475503, 0.41117864435442364]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2422.71 seconds
