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
execution time: IAR + LP analysis = 15.27 + 32.83 = 48.09 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.91 seconds, max iter: 100)

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
Binary search time: 195.51 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3356.40 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526785, upper bound: 0.6455133
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455129, upper bound: 0.6526787
time: 3.87 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.86 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.86
Output dim: 7, lower bound: -0.6526785, upper bound: 0.6455133
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.86
Output dim: 7, lower bound: -0.6455129, upper bound: 0.6526787

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6257083, 1.6286223
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6433263, 1.6469893
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6301012, 1.6324282
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2399104, 1.2393155
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.6059511, 1.6081853
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3698535, 1.3724010
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6619697, 1.6645637
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0911441, 1.0877653
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3393960, 1.3434852
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4869320, 1.4820242

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526754, upper bound: 0.6446792
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6518443, upper bound: 0.6455097
time: 3.96 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6280715, 1.6257081
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6462874, 1.6433260
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6319866, 1.6301017
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2393153, 1.2397928
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.6077554, 1.6059515
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3719206, 1.3698535
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6640763, 1.6619698
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0877652, 1.0905013
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3427110, 1.3393960
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4820244, 1.4859924

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6438767, upper bound: 0.6526768
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455088, upper bound: 0.6510446
time: 3.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.58 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.58
Output dim: 7, lower bound: -0.6526754, upper bound: 0.6446792
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.58
Output dim: 7, lower bound: -0.6518443, upper bound: 0.6455097
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.58
Output dim: 7, lower bound: -0.6438767, upper bound: 0.6526768
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.58
Output dim: 7, lower bound: -0.6455088, upper bound: 0.6510446

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6138542, 1.6280046
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6381979, 1.6467159
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6209574, 1.6319478
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2303979, 1.2388079
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.6055713, 1.6011884
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3673515, 1.3722683
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6521497, 1.6640466
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0910678, 1.0860637
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3391569, 1.3389895
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4867504, 1.4789343

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526741, upper bound: 0.6437411
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6517383, upper bound: 0.6446772
time: 4.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6250904, 1.6167684
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6430526, 1.6418612
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6296215, 1.6232839
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2394030, 1.2298027
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5989547, 1.6078048
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3697205, 1.3698995
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6614528, 1.6547437
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0894427, 1.0876887
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3349001, 1.3432459
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4838417, 1.4818430

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6516174, upper bound: 0.6452078
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6515402, upper bound: 0.6452827
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6293430, 1.6275647
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6348622, 1.6194069
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6068335, 1.6157882
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2228651, 1.2281199
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5879061, 1.5769801
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3581846, 1.3486710
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6618114, 1.6683443
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0895789, 1.0915357
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3415294, 1.3385580
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4373391, 1.4182739

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6438709, upper bound: 0.6500782
time: 3.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6413751, upper bound: 0.6526705
time: 4.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6299281, 1.6269794
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6223691, 1.6318998
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6176734, 1.6049485
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2276430, 1.2233422
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5787838, 1.5861024
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3507383, 1.3561172
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6704507, 1.6597049
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0888002, 1.0923145
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3418727, 1.3382146
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4143057, 1.4413071

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455026, upper bound: 0.6485405
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6429102, upper bound: 0.6510389
time: 3.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -0.6526741, upper bound: 0.6437411
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -0.6517383, upper bound: 0.6446772
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -0.6516174, upper bound: 0.6452078
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -0.6515402, upper bound: 0.6452827
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -0.6438709, upper bound: 0.6500782
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -0.6413751, upper bound: 0.6526705
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -0.6455026, upper bound: 0.6485405
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 7, lower bound: -0.6429102, upper bound: 0.6510389

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6131890, 1.6319735
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6364050, 1.6435895
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6118202, 1.6309509
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2315650, 1.2395812
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5818172, 1.5900624
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3453999, 1.3412733
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6395068, 1.6650051
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0905068, 1.0824542
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3488421, 1.3408689
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4723277, 1.4569769

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526735, upper bound: 0.6432328
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6521911, upper bound: 0.6437399
time: 4.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6178234, 1.6273391
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6350713, 1.6449232
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6199603, 1.6228108
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2311711, 1.2399750
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5944448, 1.5774348
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3363571, 1.3503160
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6531081, 1.6514033
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0874581, 1.0855026
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3410358, 1.3486753
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4647930, 1.4645119

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6514967, upper bound: 0.6443736
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6514166, upper bound: 0.6444504
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6250899, 1.6167679
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6430516, 1.6418560
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6296167, 1.6232831
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2394001, 1.2298021
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5989537, 1.6077995
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3697200, 1.3698978
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6614499, 1.6547399
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0894427, 1.0876882
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3348997, 1.3432463
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4838390, 1.4818368

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6516159, upper bound: 0.6442536
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6506643, upper bound: 0.6452038
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6250904, 1.6167679
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6430526, 1.6418598
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6296206, 1.6232839
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2394025, 1.2298027
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5989547, 1.6078038
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3697205, 1.3698988
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6614528, 1.6547408
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0894423, 1.0876887
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3349001, 1.3432455
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4838417, 1.4818404

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6513651, upper bound: 0.6450272
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6513579, upper bound: 0.6452789
time: 5.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6313748, 1.6232109
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6278102, 1.6227055
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6072092, 1.6149969
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2114816, 1.2334504
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5916367, 1.5689983
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3563907, 1.3495026
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6639502, 1.6637533
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0897686, 1.0911274
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3375239, 1.3404390
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4403949, 1.4117720

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6436482, upper bound: 0.6497698
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6435722, upper bound: 0.6498505
time: 3.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6249890, 1.6275647
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6348622, 1.6123555
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6060419, 1.6157882
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2228651, 1.2167364
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5799241, 1.5769801
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3581846, 1.3468771
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6572201, 1.6683443
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0891702, 1.0915357
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3415294, 1.3345525
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4308372, 1.4182739

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6413711, upper bound: 0.6517330
time: 3.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6404368, upper bound: 0.6526693
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6319599, 1.6226256
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6153171, 1.6351984
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6180487, 1.6041572
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2162595, 1.2286727
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5825148, 1.5781205
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3489444, 1.3569489
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6725895, 1.6551141
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0889900, 1.0919062
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3378673, 1.3400955
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4173617, 1.4348054

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454995, upper bound: 0.6477097
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6446684, upper bound: 0.6485374
time: 3.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6255746, 1.6269794
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6223691, 1.6248486
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6168818, 1.6049485
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2276430, 1.2119589
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5708017, 1.5861024
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3507383, 1.3543234
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6658595, 1.6597049
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0883913, 1.0923145
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3418727, 1.3342091
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4078038, 1.4413071

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6429095, upper bound: 0.6505680
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6424225, upper bound: 0.6510381
time: 3.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6526735, upper bound: 0.6432328
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6521911, upper bound: 0.6437399
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6514967, upper bound: 0.6443736
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6514166, upper bound: 0.6444504
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6516159, upper bound: 0.6442536
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6506643, upper bound: 0.6452038
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6513651, upper bound: 0.6450272
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6513579, upper bound: 0.6452789
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6436482, upper bound: 0.6497698
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6435722, upper bound: 0.6498505
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6413711, upper bound: 0.6517330
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6404368, upper bound: 0.6526693
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6454995, upper bound: 0.6477097
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6446684, upper bound: 0.6485374
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6429095, upper bound: 0.6505680
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 7, lower bound: -0.6424225, upper bound: 0.6510381

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6139581, 1.6287038
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6408167, 1.6250148
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5819058, 1.6380467
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2321889, 1.2369025
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5915928, 1.5487924
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3484392, 1.3284581
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6424146, 1.6527148
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0915534, 1.0780170
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3498950, 1.3363903
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4739480, 1.4501472

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6510368, upper bound: 0.6432282
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526693, upper bound: 0.6416583
time: 3.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6099193, 1.6319735
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6178308, 1.6435895
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6118202, 1.6010361
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2288864, 1.2395812
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5405474, 1.5900624
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3325844, 1.3412733
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6272163, 1.6650051
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0860696, 1.0824542
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3443637, 1.3408689
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4654980, 1.4569769

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6521850, upper bound: 0.6411274
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6495873, upper bound: 0.6437338
time: 5.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6178229, 1.6273389
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6350698, 1.6449177
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6199560, 1.6228099
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2311683, 1.2399744
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5944438, 1.5774293
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3363557, 1.3503141
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6531057, 1.6513995
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0874586, 1.0855024
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3410354, 1.3486757
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4647911, 1.4645057

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6498085, upper bound: 0.6443714
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6514921, upper bound: 0.6428052
time: 4.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6178234, 1.6273389
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6350713, 1.6449218
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6199598, 1.6228108
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2311707, 1.2399750
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5944448, 1.5774336
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3363571, 1.3503151
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6531081, 1.6514004
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0874579, 1.0855026
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3410358, 1.3486750
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4647930, 1.4645095

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6497268, upper bound: 0.6444487
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6514120, upper bound: 0.6428837
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6244242, 1.6207371
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6412582, 1.6387293
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6204796, 1.6222861
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2405672, 1.2305754
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5752001, 1.5966730
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3477674, 1.3389025
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6488066, 1.6556983
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0888820, 1.0840790
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3445859, 1.3451257
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4694169, 1.4598799

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6500468, upper bound: 0.6442491
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6516117, upper bound: 0.6425641
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6290591, 1.6161027
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6399245, 1.6400630
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6286201, 1.6141460
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2401733, 1.2309692
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5878272, 1.5840455
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3387246, 1.3479452
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6624088, 1.6420965
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0858335, 1.0871274
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3367791, 1.3529321
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4618824, 1.4674144

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6506580, upper bound: 0.6426038
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6480492, upper bound: 0.6451978
time: 3.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6258605, 1.6134982
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6474628, 1.6232846
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5997057, 1.6303792
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2400262, 1.2271240
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.6087306, 1.5665345
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3727601, 1.3570830
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6643608, 1.6424506
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0904896, 1.0832517
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3359523, 1.3387667
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4854612, 1.4750106

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6513590, upper bound: 0.6424234
time: 8.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6487597, upper bound: 0.6450232
time: 3.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6218212, 1.6167679
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6244774, 1.6418598
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6296206, 1.5933695
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2367237, 1.2298027
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5576851, 1.6078038
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3569052, 1.3698988
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6491630, 1.6547408
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0850055, 1.0876887
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3304210, 1.3432455
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4770122, 1.4818404

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6513563, upper bound: 0.6443274
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6503915, upper bound: 0.6452800
time: 4.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6313744, 1.6232107
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6278088, 1.6226997
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6072049, 1.6149960
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2114789, 1.2334499
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5916357, 1.5689933
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3563907, 1.3495011
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6639473, 1.6637496
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0897691, 1.0911268
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3375235, 1.3404392
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4403925, 1.4117663

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6436468, upper bound: 0.6488025
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6426416, upper bound: 0.6497700
time: 3.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6313748, 1.6232104
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6278102, 1.6227038
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6072083, 1.6149969
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2114813, 1.2334504
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5916367, 1.5689976
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3563907, 1.3495021
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6639502, 1.6637506
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0897684, 1.0911274
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3375239, 1.3404384
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4403949, 1.4117699

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6434035, upper bound: 0.6495902
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6433963, upper bound: 0.6498469
time: 3.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6413680, upper bound: 0.6508976
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405405, upper bound: 0.6517300
time: 3.63 seconds

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

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6404337, upper bound: 0.6518351
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6396035, upper bound: 0.6526661
time: 3.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6201057, 1.6220076
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6101887, 1.6349266
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6089053, 1.6036770
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2067480, 1.2281609
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5821316, 1.5711223
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3464429, 1.3568161
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6627686, 1.6545964
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0889130, 1.0902046
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3376279, 1.3355999
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4171815, 1.4317153

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6452724, upper bound: 0.6473991
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6451951, upper bound: 0.6474786
time: 3.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6313486, 1.6107714
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6150439, 1.6300702
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6175699, 1.5950131
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2157531, 1.2191612
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5755165, 1.5777385
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3488119, 1.3544481
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6720774, 1.6452934
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0872889, 1.0918295
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3333716, 1.3398575
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4142714, 1.4346240

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6446670, upper bound: 0.6476015
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6437299, upper bound: 0.6485359
time: 3.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6263442, 1.6237097
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6267796, 1.6062729
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5869675, 1.6120434
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2282665, 1.2092803
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5805774, 1.5448327
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3537769, 1.3415072
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6687694, 1.6474153
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0894380, 1.0878773
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3429246, 1.3297303
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4094226, 1.4344771

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6429065, upper bound: 0.6497958
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420740, upper bound: 0.6505645
time: 3.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6223049, 1.6269794
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6037936, 1.6248486
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6168818, 1.5750337
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2249639, 1.2119589
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5295324, 1.5861024
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3379221, 1.3543234
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6535697, 1.6597049
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0839539, 1.0923145
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3373933, 1.3342091
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4009731, 1.4413071

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6424209, upper bound: 0.6500553
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6414463, upper bound: 0.6510368
time: 3.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6510368, upper bound: 0.6432282
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6526693, upper bound: 0.6416583
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6521850, upper bound: 0.6411274
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6495873, upper bound: 0.6437338
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6498085, upper bound: 0.6443714
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6514921, upper bound: 0.6428052
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6497268, upper bound: 0.6444487
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6514120, upper bound: 0.6428837
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6500468, upper bound: 0.6442491
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6516117, upper bound: 0.6425641
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6506580, upper bound: 0.6426038
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6480492, upper bound: 0.6451978
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6513590, upper bound: 0.6424234
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6487597, upper bound: 0.6450232
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6513563, upper bound: 0.6443274
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6503915, upper bound: 0.6452800
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6436468, upper bound: 0.6488025
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6426416, upper bound: 0.6497700
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6434035, upper bound: 0.6495902
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6433963, upper bound: 0.6498469
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6413680, upper bound: 0.6508976
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6405405, upper bound: 0.6517300
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6404337, upper bound: 0.6518351
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6396035, upper bound: 0.6526661
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6452724, upper bound: 0.6473991
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6451951, upper bound: 0.6474786
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6446670, upper bound: 0.6476015
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6437299, upper bound: 0.6485359
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6429065, upper bound: 0.6497958
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6420740, upper bound: 0.6505645
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6424209, upper bound: 0.6500553
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.47
Output dim: 7, lower bound: -0.6414463, upper bound: 0.6510368

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6152282, 1.6305585
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6293592, 1.6010780
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5567131, 1.6236687
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2157393, 1.2252306
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5717809, 1.5198586
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3346729, 1.3072646
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6401072, 1.6590288
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0933676, 1.0790526
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3487148, 1.3355536
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4291837, 1.3823931

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6510311, upper bound: 0.6406133
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6485328, upper bound: 0.6432220
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6158137, 1.6299739
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6168799, 1.6135712
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.5675526, 1.6128540
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2205172, 1.2204529
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5626590, 1.5289845
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3272457, 1.3147109
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6487484, 1.6504078
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0925889, 1.0798314
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3490591, 1.3352101
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4061944, 1.4054265

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6524388, upper bound: 0.6416542
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6521900, upper bound: 0.6416581
time: 3.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.6119506, 1.6276200
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.6107798, 1.6468902
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.6121964, 1.6002452
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.2175043, 1.2449080
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.5442760, 1.5820788
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.3307910, 1.3421047
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.6293550, 1.6604143
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -1.0862586, 1.0820463
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.3403587, 1.3427503
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.4685562, 1.4504747

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6521849, upper bound: 0.6406186
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6521809, upper bound: 0.6408801
time: 6.11 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.77
Output dim: 7, lower bound: -0.6510311, upper bound: 0.6406133
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.77
Output dim: 7, lower bound: -0.6485328, upper bound: 0.6432220
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.77
Output dim: 7, lower bound: -0.6524388, upper bound: 0.6416542
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.77
Output dim: 7, lower bound: -0.6521900, upper bound: 0.6416581
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.77
Output dim: 7, lower bound: -0.6521849, upper bound: 0.6406186
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.77
Output dim: 7, lower bound: -0.6521809, upper bound: 0.6408801
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6495873, upper bound: 0.6437338
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6498085, upper bound: 0.6443714
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6514921, upper bound: 0.6428052
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6497268, upper bound: 0.6444487
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6514120, upper bound: 0.6428837
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6500468, upper bound: 0.6442491
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6516117, upper bound: 0.6425641
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6506580, upper bound: 0.6426038
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6480492, upper bound: 0.6451978
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6513590, upper bound: 0.6424234
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6487597, upper bound: 0.6450232
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6513563, upper bound: 0.6443274
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6503915, upper bound: 0.6452800
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6436468, upper bound: 0.6488025
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6426416, upper bound: 0.6497700
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6434035, upper bound: 0.6495902
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6433963, upper bound: 0.6498469
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6413680, upper bound: 0.6508976
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6405405, upper bound: 0.6517300
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6404337, upper bound: 0.6518351
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6396035, upper bound: 0.6526661
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6452724, upper bound: 0.6473991
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6451951, upper bound: 0.6474786
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6446670, upper bound: 0.6476015
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6437299, upper bound: 0.6485359
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6429065, upper bound: 0.6497958
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6420740, upper bound: 0.6505645
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6424209, upper bound: 0.6500553
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.77
Output dim: 7, lower bound: -0.6414463, upper bound: 0.6510368
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.090501308441162
rel_dist={7: [-0.6526852532184364, 0.6526854163544584]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732660, upper bound: 0.4741530
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741534, upper bound: 0.4732655
time: 3.46 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.92 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.92
Output dim: 7, lower bound: -0.4732660, upper bound: 0.4741530
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.92
Output dim: 7, lower bound: -0.4741534, upper bound: 0.4732655

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3406253, 1.3409597
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4258325, 1.4186938
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4017496, 1.4079437
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0393937, 1.0421237
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3260920, 1.3208795
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1388218, 1.1345668
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3637252, 1.3686618
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9806645, 0.9802194
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1226072, 1.1228034
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2894337, 1.2762721

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732642, upper bound: 0.4735145
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726245, upper bound: 0.4741511
time: 5.12 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3409595, 1.3406253
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4186938, 1.4258325
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4079437, 1.4017496
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0421238, 1.0393937
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3208792, 1.3260921
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1345665, 1.1388216
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3686619, 1.3637251
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9802194, 0.9806644
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1228034, 1.1226072
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2762718, 1.2894337

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741514, upper bound: 0.4713032
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4721938, upper bound: 0.4732632
time: 7.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.88
Output dim: 7, lower bound: -0.4732642, upper bound: 0.4735145
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.88
Output dim: 7, lower bound: -0.4726245, upper bound: 0.4741511
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.88
Output dim: 7, lower bound: -0.4741514, upper bound: 0.4713032
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.88
Output dim: 7, lower bound: -0.4721938, upper bound: 0.4732632

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3396640, 1.3376900
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4203911, 1.4001181
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3718352, 1.3991771
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0386024, 1.0394452
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3139911, 1.2796097
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1350651, 1.1217504
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3601203, 1.3563720
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9793611, 0.9757820
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1212893, 1.1183248
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2874320, 1.2694418

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732620, upper bound: 0.4730004
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4727530, upper bound: 0.4735094
time: 5.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3373556, 1.3399982
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4072566, 1.4132531
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3929834, 1.3780293
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0367153, 1.0413325
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2848225, 1.3087784
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1260052, 1.1308103
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3514352, 1.3650575
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9762273, 0.9789158
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1181283, 1.1214855
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2826037, 1.2742701

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726236, upper bound: 0.4735566
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4720208, upper bound: 0.4741499
time: 3.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3402548, 1.3362715
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4116418, 1.4246950
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4078193, 1.4009585
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0307403, 1.0375611
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3195906, 1.3181102
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1327732, 1.1385286
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3679163, 1.3591342
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9801531, 0.9802561
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1187983, 1.1219656
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2752318, 1.2829320

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4737891, upper bound: 0.4708558
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4737053, upper bound: 0.4709395
time: 6.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3366060, 1.3399202
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4175560, 1.4187808
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4071522, 1.4016254
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0402911, 1.0280102
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3128977, 1.3248034
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1342733, 1.1370280
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3640711, 1.3629797
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9798110, 0.9805982
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1221619, 1.1186020
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2697701, 1.2883937

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4721894, upper bound: 0.4726250
time: 3.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4715488, upper bound: 0.4732617
time: 5.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.52 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -0.4732620, upper bound: 0.4730004
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -0.4727530, upper bound: 0.4735094
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -0.4726236, upper bound: 0.4735566
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -0.4720208, upper bound: 0.4741499
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -0.4737891, upper bound: 0.4708558
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -0.4737053, upper bound: 0.4709395
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -0.4721894, upper bound: 0.4726250
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 7, lower bound: -0.4715488, upper bound: 0.4732617

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3278098, 1.3322564
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4152622, 1.3977637
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3626909, 1.3949838
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0290897, 1.0350782
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3107753, 1.2726128
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1325636, 1.1206026
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3502994, 1.3518678
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9785883, 0.9740808
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1192257, 1.1138291
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2860041, 1.2663519

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732600, upper bound: 0.4710403
time: 3.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4712996, upper bound: 0.4729986
time: 4.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3342304, 1.3258358
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4180365, 1.3949895
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3676419, 1.3900330
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0342352, 1.0299323
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3069940, 1.2763937
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1339178, 1.1192491
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3556161, 1.3465518
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9776597, 0.9750093
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1167936, 1.1162612
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2843421, 1.2680142

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4727496, upper bound: 0.4684367
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4676801, upper bound: 0.4735061
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3366890, 1.3419794
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4048696, 1.4101105
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3838072, 1.3734906
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0377140, 1.0421063
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2611063, 1.2922778
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1001568, 1.0998054
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3387499, 1.3601341
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9743605, 0.9753073
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1244695, 1.1233659
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2648921, 1.2522779

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726216, upper bound: 0.4715936
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4706602, upper bound: 0.4735576
time: 3.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3393369, 1.3393316
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4041142, 1.4108727
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3884592, 1.3688529
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0374892, 1.0423313
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2683241, 1.2850622
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0950003, 1.1049724
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3465223, 1.3523722
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9726186, 0.9770494
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1200087, 1.1278269
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2606115, 1.2565839

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4720206, upper bound: 0.4735127
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4720186, upper bound: 0.4737854
time: 5.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3402548, 1.3362712
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4116404, 1.4246910
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4078164, 1.4009576
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0307384, 1.0375606
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3195896, 1.3181069
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1327727, 1.1385279
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3679135, 1.3591307
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9801528, 0.9802555
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1187980, 1.1219659
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2752295, 1.2829278

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4737852, upper bound: 0.4706587
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4735098, upper bound: 0.4706631
time: 4.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3402548, 1.3362710
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4116418, 1.4246933
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4078188, 1.4009585
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0307398, 1.0375611
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3195906, 1.3181093
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1327732, 1.1385283
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3679163, 1.3591313
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9801526, 0.9802561
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1187983, 1.1219654
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2752318, 1.2829297

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4737015, upper bound: 0.4658670
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686320, upper bound: 0.4709366
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3356447, 1.3366506
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4121151, 1.4002051
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3772378, 1.3928595
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0394998, 1.0253316
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3007965, 1.2835339
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1305170, 1.1242120
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3604665, 1.3506899
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9785075, 0.9761609
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1208439, 1.1141231
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2677679, 1.2815633

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4721884, upper bound: 0.4720215
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4715941, upper bound: 0.4726242
time: 3.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3333364, 1.3389589
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3989801, 1.4133394
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3983860, 1.3717110
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0376124, 1.0272189
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2716284, 1.3127025
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1214571, 1.1332719
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3517814, 1.3593745
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9753737, 0.9792947
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1176829, 1.1172838
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2629397, 1.2863915

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4715486, upper bound: 0.4726243
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4715466, upper bound: 0.4728974
time: 5.92 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4732600, upper bound: 0.4710403
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4712996, upper bound: 0.4729986
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4727496, upper bound: 0.4684367
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4676801, upper bound: 0.4735061
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4726216, upper bound: 0.4715936
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4706602, upper bound: 0.4735576
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4720206, upper bound: 0.4735127
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4720186, upper bound: 0.4737854
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4737852, upper bound: 0.4706587
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4735098, upper bound: 0.4706631
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4737015, upper bound: 0.4658670
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4686320, upper bound: 0.4709366
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4721884, upper bound: 0.4720215
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4715941, upper bound: 0.4726242
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4715486, upper bound: 0.4726243
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.18
Output dim: 7, lower bound: -0.4715466, upper bound: 0.4728974

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3271046, 1.3279029
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4082108, 1.3966277
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3625669, 1.3941925
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0177073, 1.0332437
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3094835, 1.2646291
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1307707, 1.1203089
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3495545, 1.3472768
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9785213, 0.9736722
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1152205, 1.1131876
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2849641, 1.2598495

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4728955, upper bound: 0.4710355
time: 8.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726227, upper bound: 0.4710370
time: 4.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3234563, 1.3315554
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4141254, 1.3907123
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3618999, 1.3948596
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0272582, 1.0236959
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3027911, 1.2713223
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1322708, 1.1188090
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3457088, 1.3511252
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9781797, 0.9740143
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1185849, 1.1098239
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2795014, 1.2653112

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4709342, upper bound: 0.4729961
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4706613, upper bound: 0.4729983
time: 4.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3318667, 1.3251374
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4150743, 1.3941207
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3657570, 1.3894777
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0340979, 1.0294549
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3051906, 1.2758665
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1318502, 1.1186373
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3535094, 1.3459275
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9768546, 0.9722733
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1134794, 1.1152837
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2831783, 1.2640460

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4723851, upper bound: 0.4684345
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4721123, upper bound: 0.4684364
time: 4.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3335319, 1.3234721
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4171677, 1.3920274
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3670864, 1.3881483
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0337579, 1.0297949
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3064675, 1.2745901
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1333060, 1.1171815
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3549914, 1.3444453
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9749236, 0.9742041
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1158159, 1.1129470
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2803741, 1.2668505

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4676792, upper bound: 0.4729020
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4670861, upper bound: 0.4735053
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3359838, 1.3376257
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3978176, 1.4089727
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3836827, 1.3726993
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0263312, 1.0402741
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2598181, 1.2842965
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0983629, 1.0995119
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3380048, 1.3555435
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9742942, 0.9748991
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1204646, 1.1227245
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2638512, 1.2457757

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726214, upper bound: 0.4709443
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726193, upper bound: 0.4712218
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3323350, 1.3412745
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4037318, 1.4030585
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3830161, 1.3733664
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0358818, 1.0307233
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2531247, 1.2909896
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0998635, 1.0980115
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3341591, 1.3593891
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9739521, 0.9752412
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1238282, 1.1193607
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2583895, 1.2512372

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4706601, upper bound: 0.4729079
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4706580, upper bound: 0.4731857
time: 6.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3393369, 1.3393288
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4041128, 1.4108682
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3884563, 1.3688521
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0374870, 1.0423306
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2683232, 1.2850587
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0950003, 1.1049712
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3465195, 1.3523605
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9726191, 0.9770495
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1200082, 1.1278255
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2606087, 1.2565763

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4720186, upper bound: 0.4715498
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4700536, upper bound: 0.4735111
time: 3.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3393369, 1.3393315
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4041142, 1.4108713
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3884583, 1.3688529
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0374882, 1.0423313
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2683241, 1.2850611
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0950003, 1.1049721
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3465223, 1.3523691
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9726188, 0.9770494
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1200087, 1.1278265
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2606115, 1.2565813

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4720165, upper bound: 0.4718229
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4700543, upper bound: 0.4737866
time: 4.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3392930, 1.3330016
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4061995, 1.4061153
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3779020, 1.3921914
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0299499, 1.0348819
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3074889, 1.2768373
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1290169, 1.1257114
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3643093, 1.3468407
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9788506, 0.9758188
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1174798, 1.1174870
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2732275, 1.2760975

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4737830, upper bound: 0.4701475
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732739, upper bound: 0.4706579
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3369851, 1.3353071
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3930645, 1.4192488
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3990498, 1.3710432
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0280597, 1.0367693
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2783203, 1.3060060
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1199570, 1.1347711
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3556237, 1.3555174
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9757161, 0.9789525
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1143191, 1.1206462
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2683992, 1.2809227

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4735076, upper bound: 0.4701520
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4729985, upper bound: 0.4706608
time: 3.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3378911, 1.3355726
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4086802, 1.4238248
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4059334, 1.4004028
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0306027, 1.0370840
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3177872, 1.3175825
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1307056, 1.1379161
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3658106, 1.3585076
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9793477, 0.9775200
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1154835, 1.1209872
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2740679, 1.2789614

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4735090, upper bound: 0.4655875
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4735043, upper bound: 0.4658633
time: 6.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3395562, 1.3339074
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4107735, 1.4217315
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.4072628, 1.3990734
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0302627, 1.0374241
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3190637, 1.3163061
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1321609, 1.1364603
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3672926, 1.3570254
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9774168, 0.9794508
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1178205, 1.1186504
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2712636, 1.2817657

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4686311, upper bound: 0.4703372
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4680331, upper bound: 0.4709364
time: 5.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3349776, 1.3386322
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4097347, 1.3970623
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3680615, 1.3883348
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0404992, 1.0261060
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2770805, 1.2670358
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1046791, 1.0932066
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3477814, 1.3457773
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9766409, 0.9725524
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1271858, 1.1160038
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2500815, 1.2595708

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4721863, upper bound: 0.4715106
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4716774, upper bound: 0.4720193
time: 4.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3376260, 1.3359840
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4089727, 1.3978176
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3726993, 1.3836827
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0402741, 1.0263309
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2842965, 1.2598180
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0995121, 1.0983629
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3555434, 1.3380048
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9748991, 0.9742942
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1227245, 1.1204646
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2457755, 1.2638512

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4715920, upper bound: 0.4721128
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4710832, upper bound: 0.4726219
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3333364, 1.3389559
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3989792, 1.4133346
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3983831, 1.3717101
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0376108, 1.0272185
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2716269, 1.3126991
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1214571, 1.1332707
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3517780, 1.3593628
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9753742, 0.9792945
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1176827, 1.1172826
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2629375, 1.2863843

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4715476, upper bound: 0.4720239
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4709407, upper bound: 0.4726234
time: 3.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3333364, 1.3389585
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3989801, 1.4133377
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3983850, 1.3717110
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0376120, 1.0272189
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2716284, 1.3127015
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1214571, 1.1332717
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3517814, 1.3593714
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9753737, 0.9792947
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1176829, 1.1172836
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2629397, 1.2863891

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4715432, upper bound: 0.4678243
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4664737, upper bound: 0.4728937
time: 4.07 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4728955, upper bound: 0.4710355
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4726227, upper bound: 0.4710370
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4709342, upper bound: 0.4729961
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4706613, upper bound: 0.4729983
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4723851, upper bound: 0.4684345
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4721123, upper bound: 0.4684364
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4676792, upper bound: 0.4729020
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4670861, upper bound: 0.4735053
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4726214, upper bound: 0.4709443
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4726193, upper bound: 0.4712218
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4706601, upper bound: 0.4729079
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4706580, upper bound: 0.4731857
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4720186, upper bound: 0.4715498
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4700536, upper bound: 0.4735111
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4720165, upper bound: 0.4718229
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4700543, upper bound: 0.4737866
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4737830, upper bound: 0.4701475
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4732739, upper bound: 0.4706579
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4735076, upper bound: 0.4701520
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4729985, upper bound: 0.4706608
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4735090, upper bound: 0.4655875
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4735043, upper bound: 0.4658633
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4686311, upper bound: 0.4703372
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4680331, upper bound: 0.4709364
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4721863, upper bound: 0.4715106
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4716774, upper bound: 0.4720193
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4715920, upper bound: 0.4721128
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4710832, upper bound: 0.4726219
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4715476, upper bound: 0.4720239
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4709407, upper bound: 0.4726234
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4715432, upper bound: 0.4678243
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -0.4664737, upper bound: 0.4728937

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3271046, 1.3279028
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4082098, 1.3966241
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3625636, 1.3941917
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0177083, 1.0332431
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3094826, 1.2646255
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1307697, 1.1203079
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3495522, 1.3472736
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9785225, 0.9736724
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1152203, 1.1131876
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2849619, 1.2598453

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4728921, upper bound: 0.4659624
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4678226, upper bound: 0.4710319
time: 4.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.3271046, 1.3279027
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.4082108, 1.3966265
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3625660, 1.3941925
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -1.0177069, 1.0332437
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.3094835, 1.2646281
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.1307707, 1.1203084
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.3495545, 1.3472741
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9785216, 0.9736722
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.1152205, 1.1131871
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2849641, 1.2598474

Time for backsubstitution: 14.12 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=0.9791851043701172
rel_dist={7: [-0.4741558180834273, 0.47415573392575094]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111777, upper bound: 0.4106793
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106827, upper bound: 0.4111774
time: 6.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.30 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.30
Output dim: 7, lower bound: -0.4111777, upper bound: 0.4106793
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.30
Output dim: 7, lower bound: -0.4106827, upper bound: 0.4111774

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2415764, 1.2398453
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3659959, 1.3561449
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3286266, 1.3444884
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9932647, 0.9918491
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2444901, 1.2226138
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0776708, 1.0708759
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2608511, 1.2543380
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9399924, 0.9376421
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0487068, 1.0463362
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2934480, 1.2898269

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111756, upper bound: 0.4070109
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075088, upper bound: 0.4106774
time: 4.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2398455, 1.2415764
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3561449, 1.3659961
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3444881, 1.3286269
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9918489, 0.9932646
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2226138, 1.2444903
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0708759, 1.0776708
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2543385, 1.2608515
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9376421, 0.9399924
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0463362, 1.0487068
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2898269, 1.2934480

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106797, upper bound: 0.4106812
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106809, upper bound: 0.4109645
time: 5.22 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.44 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.44
Output dim: 7, lower bound: -0.4111756, upper bound: 0.4070109
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.44
Output dim: 7, lower bound: -0.4075088, upper bound: 0.4106774
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.44
Output dim: 7, lower bound: -0.4106797, upper bound: 0.4106812
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.44
Output dim: 7, lower bound: -0.4106809, upper bound: 0.4109645

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2392128, 1.2387304
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3630342, 1.3547535
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3267422, 1.3436003
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9930425, 0.9913719
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2426867, 1.2217674
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0756037, 1.0699005
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2587454, 1.2533433
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9387047, 0.9349062
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0453920, 1.0447741
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2915831, 1.2858586

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111738, upper bound: 0.4056258
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4056292, upper bound: 0.4070118
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2404616, 1.2374816
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3646040, 1.3531833
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3277388, 1.3426032
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9927874, 0.9916270
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2436442, 1.2208102
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0766954, 1.0688088
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2598565, 1.2522317
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9372566, 0.9363544
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0471447, 1.0430214
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2894797, 1.2879620

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075081, upper bound: 0.4102130
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4070361, upper bound: 0.4106765
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2398450, 1.2415740
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3561440, 1.3659928
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3444862, 1.3286259
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9918473, 0.9932638
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2226129, 1.2444873
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0708754, 1.0776696
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2543347, 1.2608421
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9376423, 0.9399924
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0463357, 1.0487056
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2898245, 1.2934420

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100131, upper bound: 0.4106791
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106779, upper bound: 0.4100144
time: 4.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2398455, 1.2415761
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3561449, 1.3659952
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3444872, 1.3286269
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9918485, 0.9932646
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2226138, 1.2444891
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0708759, 1.0776703
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2543385, 1.2608485
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9376421, 0.9399924
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0463362, 1.0487063
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2898269, 1.2934458

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106763, upper bound: 0.4095786
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4092952, upper bound: 0.4109625
time: 6.07 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 7, lower bound: -0.4111738, upper bound: 0.4056258
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 26.21
Output dim: 7, lower bound: -0.4056292, upper bound: 0.4070118
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 7, lower bound: -0.4075081, upper bound: 0.4102130
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 7, lower bound: -0.4070361, upper bound: 0.4106765
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 7, lower bound: -0.4100131, upper bound: 0.4106791
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 7, lower bound: -0.4106779, upper bound: 0.4100144
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 7, lower bound: -0.4106763, upper bound: 0.4095786
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 7, lower bound: -0.4092952, upper bound: 0.4109625

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2375956, 1.2343763
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3559828, 1.3521371
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3264508, 1.3428087
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9816589, 0.9871516
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2397246, 1.2137853
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0738101, 1.0692322
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2570386, 1.2487525
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9385529, 0.9344977
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0413868, 1.0432916
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2891777, 1.2793572

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4109611, upper bound: 0.4056259
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106777, upper bound: 0.4056289
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2397959, 1.2388021
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3620510, 1.3500583
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3186021, 1.3369551
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9937298, 0.9924005
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2198896, 1.2024678
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0495765, 1.0378144
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2472138, 1.2454177
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9349537, 0.9327450
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0523694, 1.0449007
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2707520, 1.2660046

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4072940, upper bound: 0.4102128
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4070120, upper bound: 0.4102131
time: 4.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2417822, 1.2368159
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3614793, 1.3506300
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3220906, 1.3334661
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9935610, 0.9925693
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2253017, 1.1970561
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0457010, 1.0416896
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2530432, 1.2395884
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9336472, 0.9340514
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0490236, 1.0482461
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2675223, 1.2692337

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058710, upper bound: 0.4106755
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4065350, upper bound: 0.4100131
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2411160, 1.2430959
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3375788, 1.3420734
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3193331, 1.3081188
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9753976, 0.9788615
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1975498, 1.2155153
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0528841, 1.0564871
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2520709, 1.2622805
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9390106, 0.9410269
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0451541, 1.0476711
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2319767, 1.2257230

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100124, upper bound: 0.4102146
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4095492, upper bound: 0.4106819
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2413669, 1.2428451
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3322248, 1.3474271
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3239779, 1.3034730
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9774449, 0.9768139
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1936407, 1.2194247
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0496929, 1.0596783
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2557731, 1.2585773
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9386768, 0.9413607
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0453014, 1.0475240
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2221055, 1.2355945

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106761, upper bound: 0.4096300
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4102936, upper bound: 0.4100126
time: 3.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2382278, 1.2372220
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3490934, 1.3633788
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3441958, 1.3278356
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9804649, 0.9890442
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2196517, 1.2365069
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0690825, 1.0770023
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2526312, 1.2562578
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9374905, 0.9395840
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0423307, 1.0472238
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2874212, 1.2869437

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100096, upper bound: 0.4095769
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106745, upper bound: 0.4089118
time: 4.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2354910, 1.2399585
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3535290, 1.3589432
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3436961, 1.3283358
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9876282, 0.9818810
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2146316, 1.2415267
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0702078, 1.0758770
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2497473, 1.2591419
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9372339, 0.9398407
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0448534, 1.0447011
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2833250, 1.2910399

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4092905, upper bound: 0.4105781
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4089107, upper bound: 0.4109605
time: 5.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4109611, upper bound: 0.4056259
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4106777, upper bound: 0.4056289
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4072940, upper bound: 0.4102128
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4070120, upper bound: 0.4102131
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4058710, upper bound: 0.4106755
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4065350, upper bound: 0.4100131
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4100124, upper bound: 0.4102146
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4095492, upper bound: 0.4106819
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4106761, upper bound: 0.4096300
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4102936, upper bound: 0.4100126
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4100096, upper bound: 0.4095769
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4106745, upper bound: 0.4089118
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4092905, upper bound: 0.4105781
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.63
Output dim: 7, lower bound: -0.4089107, upper bound: 0.4109605

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2375946, 1.2343761
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3559809, 1.3521340
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3264484, 1.3428080
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9816592, 0.9871508
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2397232, 1.2137825
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0738099, 1.0692317
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2570357, 1.2487491
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9385531, 0.9344974
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0413864, 1.0432916
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2891750, 1.2793531

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4109592, upper bound: 0.4052398
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4105766, upper bound: 0.4056225
time: 5.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2375956, 1.2343760
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3559828, 1.3521357
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3264499, 1.3428087
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9816582, 0.9871516
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2397246, 1.2137843
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0738101, 1.0692320
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2570386, 1.2487495
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9385526, 0.9344977
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0413868, 1.0432913
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2891777, 1.2793546

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106758, upper bound: 0.4052412
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4102932, upper bound: 0.4056248
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2397959, 1.2388022
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3620496, 1.3500555
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3185997, 1.3369544
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9937298, 0.9923997
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2198887, 1.2024646
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0495760, 1.0378134
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2472110, 1.2454145
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9349539, 0.9327446
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0523691, 1.0449007
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2707498, 1.2660007

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4072922, upper bound: 0.4098270
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4066285, upper bound: 0.4102096
time: 4.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2397959, 1.2388021
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3620510, 1.3500571
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3186011, 1.3369551
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9937289, 0.9924005
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2198896, 1.2024665
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0495765, 1.0378139
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2472138, 1.2454149
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9349535, 0.9327450
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0523694, 1.0449004
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2707520, 1.2660024

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4063454, upper bound: 0.4102107
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4070095, upper bound: 0.4095465
time: 5.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2430522, 1.2383368
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3428960, 1.3266931
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.2968979, 1.3129184
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9771109, 0.9781668
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2002785, 1.1681221
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0276988, 1.0204961
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2507362, 1.2409841
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9350164, 0.9350868
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0478437, 1.0472136
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2096403, 1.2014799

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4061534, upper bound: 0.4106730
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058699, upper bound: 0.4106778
time: 3.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2433028, 1.2380860
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3375425, 1.3320420
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3015327, 1.3082733
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9791584, 0.9761192
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1963675, 1.1720315
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0245075, 1.0236793
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2544317, 1.2372816
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9346826, 0.9354205
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0479908, 1.0470660
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.1997688, 1.2113326

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4061507, upper bound: 0.4096289
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4066499, upper bound: 0.4100087
time: 4.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2404497, 1.2444154
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3350024, 1.3389311
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3101563, 1.3024206
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9763398, 0.9796350
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1738341, 1.1972109
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0257463, 1.0254819
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2393851, 1.2554166
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9367087, 0.9374186
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0503798, 1.0495512
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2131948, 1.2037306

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100105, upper bound: 0.4088494
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4086266, upper bound: 0.4102158
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2424359, 1.2424295
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3344364, 1.3395028
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3136454, 1.2989423
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9761713, 0.9798038
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1792474, 1.1917992
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0218790, 1.0293574
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2452145, 1.2495952
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9354024, 0.9387251
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0470343, 1.0528972
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2099843, 1.2069600

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4095471, upper bound: 0.4092931
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4081863, upper bound: 0.4106766
time: 4.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2295129, 1.2358067
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3270960, 1.3443794
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3148336, 1.2980418
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9679322, 0.9711604
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1894794, 1.2124279
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0471911, 1.0581918
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2459531, 1.2527444
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9376724, 0.9396596
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0426300, 1.0430284
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2202623, 1.2325046

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4098318, upper bound: 0.4091694
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4102114, upper bound: 0.4096296
time: 4.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2343283, 1.2309911
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3291769, 1.3422987
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3185468, 1.2943287
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9717915, 0.9673010
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1866441, 1.2152635
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0482063, 1.0571766
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2499399, 1.2487574
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9369760, 0.9403561
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0408058, 1.0448527
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2190156, 1.2337511

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6183
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4089090, upper bound: 0.4086300
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4089077, upper bound: 0.4100106
time: 4.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2394991, 1.2387443
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3305283, 1.3394597
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3190432, 1.3073282
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9640150, 0.9746416
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1945896, 1.2075355
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0510910, 1.0558197
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2503667, 1.2576959
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9388587, 0.9406186
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0411491, 1.0461895
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2295735, 1.2192247

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4602
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4602

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100077, upper bound: 0.4091945
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4096251, upper bound: 0.4095771
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2397504, 1.2384934
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3251743, 1.3448131
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3236885, 1.3026829
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9660625, 0.9725940
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.1906800, 1.2114450
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0478997, 1.0590110
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2540689, 1.2539928
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9385250, 0.9409524
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0412965, 1.0460422
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2197020, 1.2290962

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4602

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106724, upper bound: 0.4052461
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4070056, upper bound: 0.4089098
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2236376, 1.2329227
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3484011, 1.3558960
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3345523, 1.3229051
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9781165, 0.9762288
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2104688, 1.2345282
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0677063, 1.0743906
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2399273, 1.2533109
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9362288, 0.9381392
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0421824, 1.0402052
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2814813, 1.2879496

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6153
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4092898, upper bound: 0.4101132
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4088466, upper bound: 0.4105775
time: 7.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2284529, 1.2281044
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3504829, 1.3538156
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3382654, 1.3191919
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9819736, 0.9723694
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2076330, 1.2373632
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0687211, 1.0733755
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2439146, 1.2493217
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9355323, 0.9388353
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0403578, 1.0420294
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2802346, 1.2891967

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4082412, upper bound: 0.4109623
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4089061, upper bound: 0.4102940
time: 5.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4109592, upper bound: 0.4052398
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4105766, upper bound: 0.4056225
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4106758, upper bound: 0.4052412
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4102932, upper bound: 0.4056248
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4072922, upper bound: 0.4098270
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4066285, upper bound: 0.4102096
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4063454, upper bound: 0.4102107
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4070095, upper bound: 0.4095465
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4061534, upper bound: 0.4106730
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4058699, upper bound: 0.4106778
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4061507, upper bound: 0.4096289
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4066499, upper bound: 0.4100087
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4100105, upper bound: 0.4088494
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4086266, upper bound: 0.4102158
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4095471, upper bound: 0.4092931
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4081863, upper bound: 0.4106766
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4098318, upper bound: 0.4091694
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4102114, upper bound: 0.4096296
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4089090, upper bound: 0.4086300
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4089077, upper bound: 0.4100106
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4100077, upper bound: 0.4091945
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4096251, upper bound: 0.4095771
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4106724, upper bound: 0.4052461
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4070056, upper bound: 0.4089098
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4092898, upper bound: 0.4101132
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4088466, upper bound: 0.4105775
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4082412, upper bound: 0.4109623
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4089061, upper bound: 0.4102940

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2257409, 1.2273376
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3508530, 1.3490872
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3173041, 1.3373771
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9721477, 0.9814962
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2355599, 1.2067840
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0713079, 1.0677445
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2472153, 1.2429161
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9375479, 0.9327962
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0387146, 1.0387957
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2873321, 1.2762628

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4102925, upper bound: 0.4052384
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4109574, upper bound: 0.4045729
time: 5.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2305591, 1.2225220
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3529334, 1.3470061
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3210173, 1.3336639
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9760067, 0.9776392
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2327247, 1.2096195
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0723231, 1.0667295
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2512045, 1.2389290
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9368520, 0.9334927
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0368907, 1.0406203
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2860849, 1.2775095

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4099098, upper bound: 0.4056205
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4105748, upper bound: 0.4049589
time: 3.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2257414, 1.2273375
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3508549, 1.3490889
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3173056, 1.3373778
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9721467, 0.9814970
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2355614, 1.2067858
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0713089, 1.0677447
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2472181, 1.2429165
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9375472, 0.9327965
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0387151, 1.0387955
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2873344, 1.2762642

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4100091, upper bound: 0.4052395
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106740, upper bound: 0.4045778
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2305596, 1.2225219
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3529348, 1.3470075
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3210187, 1.3336647
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9760058, 0.9776400
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2327261, 1.2096214
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0723240, 1.0667300
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2512074, 1.2389294
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9368513, 0.9334929
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0368910, 1.0406201
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2860870, 1.2775109

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4585
type: RSZ, layer: 1, pos: 6153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4585

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4096265, upper bound: 0.4056221
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4102914, upper bound: 0.4049570
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.4819937, -9.2464142, -11.4819937, -9.2464142, -1.2279420, 1.2317636
1: -6.5261440, -4.7152481, -6.5261440, -4.7152481, -1.3569207, 1.3470070
2: -6.2376757, -4.2180405, -6.2376757, -4.2180405, -1.3094554, 1.3315232
3: -5.3569956, -3.7469783, -5.3569956, -3.7469783, -0.9842169, 0.9867461
4: -7.4061127, -5.1482797, -7.4061127, -5.1482797, -1.2157269, 1.1954676
5: -10.4922609, -8.6001892, -10.4922609, -8.6001892, -1.0470741, 1.0363266
6: -17.1402016, -14.7059708, -17.1402016, -14.7059708, -1.2373900, 1.2395813
7: 5.0486193, 6.2599268, 5.0486193, 6.2599268, -0.9339492, 0.9310434
8: -6.4546914, -4.6735835, -6.4546914, -4.6735835, -1.0496979, 1.0404053
9: -5.4519515, -3.7852187, -5.4519515, -3.7852187, -1.2689059, 1.2629104

Time for backsubstitution: 14.43 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=0.9420795440673828
rel_dist={7: [-0.411179151475503, 0.41117864435442364]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2421.51 seconds
