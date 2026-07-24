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
execution time: IAR + LP analysis = 14.91 + 33.40 = 48.31 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.69 seconds, max iter: 100)

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
Binary search time: 197.64 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3354.06 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 577

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526786, upper bound: 0.6454348
time: 4.07 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526785, upper bound: 0.6526784
time: 3.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.21 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.21
Output dim: 7, lower bound: -0.6526786, upper bound: 0.6454348
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.21
Output dim: 7, lower bound: -0.6526785, upper bound: 0.6526784

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -11.4713383, -9.2479610, -11.4819937, -9.2464142, -1.6172173, 1.6279664
1: -6.5234876, -4.7154131, -6.5261440, -4.7152481, -1.6429291, 1.6445668
2: -6.2258978, -4.2193713, -6.2376757, -4.2180405, -1.6200004, 1.6319361
3: -5.3554783, -3.7503858, -5.3569956, -3.7469783, -1.2378035, 1.2380015
4: -7.4019489, -5.1485786, -7.4061127, -5.1482797, -1.6043313, 1.6075757
5: -10.4817095, -8.6020994, -10.4922609, -8.6001892, -1.3613429, 1.3716476
6: -17.1340351, -14.7069759, -17.1402016, -14.7059708, -1.6577377, 1.6638061
7: 5.0498300, 6.2543149, 5.0486193, 6.2599268, -1.0901968, 1.0847412
8: -6.4465570, -4.6751695, -6.4546914, -4.6735835, -1.3344817, 1.3426312
9: -5.4514394, -3.7926459, -5.4519515, -3.7852187, -1.4856966, 1.4785383

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6454344
time: 3.97 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6454342
time: 3.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.4845057, -9.1906204, -11.4819899, -9.2464161, -1.6312227, 1.6593393
1: -6.5281162, -4.7085695, -6.5261416, -4.7152481, -1.6567769, 1.6519260
2: -6.2393103, -4.1525426, -6.2376728, -4.2180400, -1.6334467, 1.6625708
3: -5.3599415, -3.7378778, -5.3569946, -3.7469792, -1.2425480, 1.2512515
4: -7.4078946, -5.1316462, -7.4061098, -5.1482801, -1.6118431, 1.6251973
5: -10.4926367, -8.5582094, -10.4922543, -8.6001892, -1.3727701, 1.3927522
6: -17.1419563, -14.6815882, -17.1401978, -14.7059717, -1.6664193, 1.6824174
7: 5.0200677, 6.2601023, 5.0486197, 6.2599249, -1.1087461, 1.0911949
8: -6.4556770, -4.6370978, -6.4546881, -4.6735830, -1.3442883, 1.3719316
9: -5.4970675, -3.7843614, -5.4519515, -3.7852201, -1.5275378, 1.4878089

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6526787
time: 3.90 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6526787
time: 4.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.93 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.93
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6454344
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.93
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6454342
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.93
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6526787
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.93
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6526787

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -11.4713383, -9.2479610, -11.4713383, -9.2479610, -1.6171119, 1.6171117
1: -6.5234876, -4.7154131, -6.5234876, -4.7154131, -1.6412082, 1.6412082
2: -6.2258978, -4.2193713, -6.2258978, -4.2193713, -1.6199493, 1.6199496
3: -5.3554783, -3.7503858, -5.3554783, -3.7503858, -1.2360122, 1.2360120
4: -7.4019489, -5.1485786, -7.4019489, -5.1485786, -1.6041520, 1.6041517
5: -10.4817095, -8.6020994, -10.4817095, -8.6020994, -1.3610697, 1.3610699
6: -17.1340351, -14.7069759, -17.1340351, -14.7069759, -1.6574678, 1.6574676
7: 5.0498300, 6.2543149, 5.0498300, 6.2543149, -1.0844369, 1.0844367
8: -6.4465570, -4.6751695, -6.4465570, -4.6751695, -1.3344021, 1.3344021
9: -5.4514394, -3.7926459, -5.4514394, -3.7926459, -1.4782424, 1.4782424

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454311, upper bound: 0.6422906
time: 7.24 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454286, upper bound: 0.6454280
time: 4.10 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -11.4713383, -9.2479610, -11.4845057, -9.1906204, -1.6484778, 1.6305645
1: -6.5234876, -4.7154131, -6.5281162, -4.7085695, -1.6485686, 1.6453798
2: -6.2258978, -4.2193713, -6.2393103, -4.1525426, -1.6505787, 1.6329532
3: -5.3554783, -3.7503858, -5.3599415, -3.7378778, -1.2471087, 1.2407576
4: -7.4019489, -5.1485786, -7.4078946, -5.1316462, -1.6217752, 1.6092579
5: -10.4817095, -8.6020994, -10.4926367, -8.5582094, -1.3821678, 1.3720157
6: -17.1340351, -14.7069759, -17.1419563, -14.6815882, -1.6760776, 1.6656601
7: 5.0498300, 6.2543149, 5.0200677, 6.2601023, -1.0902460, 1.1029866
8: -6.4465570, -4.6751695, -6.4556770, -4.6370978, -1.3636980, 1.3434308
9: -5.4514394, -3.7926459, -5.4970675, -3.7843614, -1.4865699, 1.5200788

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454286, upper bound: 0.6422910
time: 5.91 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454286, upper bound: 0.6454281
time: 4.88 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.4845057, -9.1906204, -11.4713383, -9.2479610, -1.6305645, 1.6484776
1: -6.5281162, -4.7085695, -6.5234876, -4.7154131, -1.6453795, 1.6485691
2: -6.2393103, -4.1525426, -6.2258978, -4.2193713, -1.6329527, 1.6505787
3: -5.3599415, -3.7378778, -5.3554783, -3.7503858, -1.2407575, 1.2471085
4: -7.4078946, -5.1316462, -7.4019489, -5.1485786, -1.6092579, 1.6217750
5: -10.4926367, -8.5582094, -10.4817095, -8.6020994, -1.3720160, 1.3821678
6: -17.1419563, -14.6815882, -17.1340351, -14.7069759, -1.6656604, 1.6760772
7: 5.0200677, 6.2601023, 5.0498300, 6.2543149, -1.1029866, 1.0902461
8: -6.4556770, -4.6370978, -6.4465570, -4.6751695, -1.3434310, 1.3636980
9: -5.4970675, -3.7843614, -5.4514394, -3.7926459, -1.5200787, 1.4865699

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454282, upper bound: 0.6495339
time: 7.13 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454282, upper bound: 0.6526716
time: 7.09 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.4845057, -9.1906204, -11.4845057, -9.1906204, -1.6619055, 1.6619055
1: -6.5281162, -4.7085695, -6.5281162, -4.7085695, -1.6592278, 1.6592278
2: -6.2393103, -4.1525426, -6.2393103, -4.1525426, -1.6635988, 1.6635985
3: -5.3599415, -3.7378778, -5.3599415, -3.7378778, -1.2544897, 1.2544897
4: -7.4078946, -5.1316462, -7.4078946, -5.1316462, -1.6234372, 1.6234373
5: -10.4926367, -8.5582094, -10.4926367, -8.5582094, -1.3931236, 1.3931236
6: -17.1419563, -14.6815882, -17.1419563, -14.6815882, -1.6842763, 1.6842766
7: 5.0200677, 6.2601023, 5.0200677, 6.2601023, -1.1088006, 1.1088008
8: -6.4556770, -4.6370978, -6.4556770, -4.6370978, -1.3727446, 1.3727446
9: -5.4970675, -3.7843614, -5.4970675, -3.7843614, -1.5234466, 1.5234463

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454282, upper bound: 0.6495339
time: 4.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454282, upper bound: 0.6526718
time: 4.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.32 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.32
Output dim: 7, lower bound: -0.6454311, upper bound: 0.6422906
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.32
Output dim: 7, lower bound: -0.6454286, upper bound: 0.6454280
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.32
Output dim: 7, lower bound: -0.6454286, upper bound: 0.6422910
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.32
Output dim: 7, lower bound: -0.6454286, upper bound: 0.6454281
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.32
Output dim: 7, lower bound: -0.6454282, upper bound: 0.6495339
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.32
Output dim: 7, lower bound: -0.6454282, upper bound: 0.6526716
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.32
Output dim: 7, lower bound: -0.6454282, upper bound: 0.6495339
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.32
Output dim: 7, lower bound: -0.6454282, upper bound: 0.6526718

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4677734, -9.2534456, -11.4713383, -9.2479610, -1.6118338, 1.6117735
1: -6.5157404, -4.7188802, -6.5234876, -4.7154131, -1.6314211, 1.6380658
2: -6.2238855, -4.2203841, -6.2258978, -4.2193713, -1.6184540, 1.6180215
3: -5.3383713, -3.7518415, -5.3554783, -3.7503858, -1.2191324, 1.2355152
4: -7.3989377, -5.1582294, -7.4019489, -5.1485786, -1.5991709, 1.5946600
5: -10.4773235, -8.6045570, -10.4817095, -8.6020994, -1.3562679, 1.3587565
6: -17.1317863, -14.7180099, -17.1340351, -14.7069759, -1.6547832, 1.6458062
7: 5.0525694, 6.2496824, 5.0498300, 6.2543149, -1.0815277, 1.0785197
8: -6.4325776, -4.6761212, -6.4465570, -4.6751695, -1.3206196, 1.3326432
9: -5.4464316, -3.8021157, -5.4514394, -3.7926459, -1.4729528, 1.4689288

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422897, upper bound: 0.6422898
time: 5.52 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422897, upper bound: 0.6422896
time: 5.77 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -11.5212736, -9.2449646, -11.4713345, -9.2479715, -1.6489296, 1.6254652
1: -6.5370526, -4.6672063, -6.5234766, -4.7154160, -1.6721489, 1.6675370
2: -6.2503891, -4.2160387, -6.2258949, -4.2193718, -1.6441586, 1.6277988
3: -5.3648047, -3.6843164, -5.3554678, -3.7503879, -1.2527888, 1.2525259
4: -7.4495029, -5.1463509, -7.4019456, -5.1485858, -1.6259174, 1.6117632
5: -10.4868412, -8.5776625, -10.4817066, -8.6021023, -1.3655736, 1.3798475
6: -17.1838722, -14.7003584, -17.1340332, -14.7069921, -1.6844397, 1.6672876
7: 5.0275087, 6.2566948, 5.0498328, 6.2543116, -1.1001821, 1.0865849
8: -6.4502249, -4.6247807, -6.4465437, -4.6751699, -1.3408289, 1.3615510
9: -5.4927726, -3.7862911, -5.4514327, -3.7926519, -1.5109913, 1.4885623

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454246, upper bound: 0.6437990
time: 3.70 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454246, upper bound: 0.6454267
time: 3.67 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.4677734, -9.2534456, -11.4845057, -9.1906204, -1.6433260, 1.6252265
1: -6.5157404, -4.7188802, -6.5281162, -4.7085695, -1.6387820, 1.6422379
2: -6.2238855, -4.2203841, -6.2393103, -4.1525426, -1.6491292, 1.6310248
3: -5.3383713, -3.7518415, -5.3599415, -3.7378778, -1.2302287, 1.2402608
4: -7.3989377, -5.1582294, -7.4078946, -5.1316462, -1.6167941, 1.5997664
5: -10.4773235, -8.6045570, -10.4926367, -8.5582094, -1.3773522, 1.3697021
6: -17.1317863, -14.7180099, -17.1419563, -14.6815882, -1.6733963, 1.6539989
7: 5.0525694, 6.2496824, 5.0200677, 6.2601023, -1.0873368, 1.0970590
8: -6.4325776, -4.6761212, -6.4556770, -4.6370978, -1.3498220, 1.3416719
9: -5.4464316, -3.8021157, -5.4970675, -3.7843614, -1.4812799, 1.5107480

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6495320, upper bound: 0.6422895
time: 7.23 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6495320, upper bound: 0.6422915
time: 3.91 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.5212736, -9.2449646, -11.4845018, -9.1906309, -1.6655257, 1.6389186
1: -6.5370526, -4.6672063, -6.5281053, -4.7085710, -1.6795096, 1.6717172
2: -6.2503891, -4.2160387, -6.2393069, -4.1525440, -1.6653628, 1.6408024
3: -5.3648047, -3.6843164, -5.3599324, -3.7378800, -1.2638853, 1.2573036
4: -7.4495029, -5.1463509, -7.4078913, -5.1316547, -1.6326230, 1.6168698
5: -10.4868412, -8.5776625, -10.4926338, -8.5582123, -1.3866842, 1.3908035
6: -17.1838722, -14.7003584, -17.1419563, -14.6816063, -1.6891832, 1.6754800
7: 5.0275087, 6.2566948, 5.0200710, 6.2600989, -1.1059954, 1.1051548
8: -6.4502249, -4.6247807, -6.4556637, -4.6370983, -1.3701332, 1.3705978
9: -5.4927726, -3.7862911, -5.4970622, -3.7843673, -1.5193243, 1.5303833

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526678, upper bound: 0.6437983
time: 3.98 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526677, upper bound: 0.6454262
time: 4.19 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.4809484, -9.1961031, -11.4713383, -9.2479610, -1.6253016, 1.6431289
1: -6.5203676, -4.7120333, -6.5234876, -4.7154131, -1.6355853, 1.6454258
2: -6.2372913, -4.1535563, -6.2258978, -4.2193713, -1.6314507, 1.6486409
3: -5.3428459, -3.7393341, -5.3554783, -3.7503858, -1.2238836, 1.2466114
4: -7.4048805, -5.1412959, -7.4019489, -5.1485786, -1.6042750, 1.6122801
5: -10.4882488, -8.5606651, -10.4817095, -8.6020994, -1.3672132, 1.3798542
6: -17.1397114, -14.6926374, -17.1340351, -14.7069759, -1.6629763, 1.6643457
7: 5.0228024, 6.2554703, 5.0498300, 6.2543149, -1.1000865, 1.0843288
8: -6.4416952, -4.6380491, -6.4465570, -4.6751695, -1.3296466, 1.3619443
9: -5.4920688, -3.7938344, -5.4514394, -3.7926459, -1.5147581, 1.4772530

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422892, upper bound: 0.6495321
time: 5.33 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422892, upper bound: 0.6495322
time: 5.96 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.5344362, -9.1876230, -11.4713345, -9.2479715, -1.6623707, 1.6557047
1: -6.5416098, -4.6603680, -6.5234766, -4.7154160, -1.6763313, 1.6701968
2: -6.2638006, -4.1492076, -6.2258949, -4.2193718, -1.6571467, 1.6544869
3: -5.3692389, -3.6718063, -5.3554678, -3.7503879, -1.2574947, 1.2562373
4: -7.4554482, -5.1294107, -7.4019456, -5.1485858, -1.6310153, 1.6293718
5: -10.4977684, -8.5337782, -10.4817066, -8.6021023, -1.3765199, 1.3880486
6: -17.1917953, -14.6749563, -17.1340332, -14.7069921, -1.6926389, 1.6859753
7: 4.9977527, 6.2624812, 5.0498328, 6.2543116, -1.1097441, 1.0923938
8: -6.4593468, -4.5867109, -6.4465437, -4.6751699, -1.3498540, 1.3712568
9: -5.5383878, -3.7780240, -5.4514327, -3.7926519, -1.5279448, 1.4968739

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454240, upper bound: 0.6510401
time: 3.80 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454240, upper bound: 0.6526699
time: 3.82 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.4809484, -9.1961031, -11.4845057, -9.1906204, -1.6567681, 1.6565568
1: -6.5203676, -4.7120333, -6.5281162, -4.7085695, -1.6494451, 1.6560919
2: -6.2372913, -4.1535563, -6.2393103, -4.1525426, -1.6621416, 1.6616606
3: -5.3428459, -3.7393341, -5.3599415, -3.7378778, -1.2376153, 1.2539928
4: -7.4048805, -5.1412959, -7.4078946, -5.1316462, -1.6184576, 1.6139429
5: -10.4882488, -8.5606651, -10.4926367, -8.5582094, -1.3883080, 1.3908100
6: -17.1397114, -14.6926374, -17.1419563, -14.6815882, -1.6815956, 1.6725450
7: 5.0228024, 6.2554703, 5.0200677, 6.2601023, -1.1059008, 1.1028725
8: -6.4416952, -4.6380491, -6.4556770, -4.6370978, -1.3588676, 1.3709910
9: -5.4920688, -3.7938344, -5.4970675, -3.7843614, -1.5181699, 1.5141289

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6423669, upper bound: 0.6495331
time: 6.10 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6423669, upper bound: 0.6495328
time: 5.36 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.5344362, -9.1876230, -11.4845018, -9.1906309, -1.6789663, 1.6691329
1: -6.5416098, -4.6603680, -6.5281053, -4.7085710, -1.6878524, 1.6780090
2: -6.2638006, -4.1492076, -6.2393069, -4.1525440, -1.6783509, 1.6675060
3: -5.3692389, -3.6718063, -5.3599324, -3.7378800, -1.2712269, 1.2619780
4: -7.4554482, -5.1294107, -7.4078913, -5.1316547, -1.6381879, 1.6310346
5: -10.4977684, -8.5337782, -10.4926338, -8.5582123, -1.3976407, 1.3990046
6: -17.1917953, -14.6749563, -17.1419563, -14.6816063, -1.6973825, 1.6941745
7: 4.9977527, 6.2624812, 5.0200710, 6.2600989, -1.1155574, 1.1109686
8: -6.4593468, -4.5867109, -6.4556637, -4.6370983, -1.3791769, 1.3803037
9: -5.5383878, -3.7780240, -5.4970622, -3.7843673, -1.5362778, 1.5337498

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455026, upper bound: 0.6510384
time: 4.15 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455026, upper bound: 0.6526704
time: 3.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.72 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6422897, upper bound: 0.6422898
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6422897, upper bound: 0.6422896
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6454246, upper bound: 0.6437990
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6454246, upper bound: 0.6454267
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6495320, upper bound: 0.6422895
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6495320, upper bound: 0.6422915
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6526678, upper bound: 0.6437983
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6526677, upper bound: 0.6454262
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6422892, upper bound: 0.6495321
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6422892, upper bound: 0.6495322
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6454240, upper bound: 0.6510401
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6454240, upper bound: 0.6526699
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6423669, upper bound: 0.6495331
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6423669, upper bound: 0.6495328
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6455026, upper bound: 0.6510384
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.72
Output dim: 7, lower bound: -0.6455026, upper bound: 0.6526704

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.4677734, -9.2534456, -11.4677734, -9.2534456, -1.6064959, 1.6064961
1: -6.5157404, -4.7188802, -6.5157404, -4.7188802, -1.6282792, 1.6282792
2: -6.2238855, -4.2203841, -6.2238855, -4.2203841, -1.6165257, 1.6165259
3: -5.3383713, -3.7518415, -5.3383713, -3.7518415, -1.2186353, 1.2186354
4: -7.3989377, -5.1582294, -7.3989377, -5.1582294, -1.5896788, 1.5896792
5: -10.4773235, -8.6045570, -10.4773235, -8.6045570, -1.3539548, 1.3539548
6: -17.1317863, -14.7180099, -17.1317863, -14.7180099, -1.6431217, 1.6431222
7: 5.0525694, 6.2496824, 5.0525694, 6.2496824, -1.0756106, 1.0756105
8: -6.4325776, -4.6761212, -6.4325776, -4.6761212, -1.3188605, 1.3188605
9: -5.4464316, -3.8021157, -5.4464316, -3.8021157, -1.4636385, 1.4636388

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405202, upper bound: 0.6422896
time: 4.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422856, upper bound: 0.6422898
time: 3.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.4677734, -9.2534456, -11.5207462, -9.2449646, -1.6167159, 1.6434200
1: -6.5157404, -4.7188802, -6.5369616, -4.6674967, -1.6576393, 1.6525524
2: -6.2238855, -4.2203841, -6.2503123, -4.2161674, -1.6217217, 1.6420805
3: -5.3383713, -3.7518415, -5.3647890, -3.6852601, -1.2349401, 1.2469140
4: -7.3989377, -5.1582294, -7.4483662, -5.1463513, -1.6030226, 1.6160319
5: -10.4773235, -8.6045570, -10.4868364, -8.5781603, -1.3747272, 1.3624272
6: -17.1317863, -14.7180099, -17.1823635, -14.7004156, -1.6623721, 1.6720198
7: 5.0525694, 6.2496824, 5.0279980, 6.2566948, -1.0834804, 1.0938606
8: -6.4325776, -4.6761212, -6.4501886, -4.6278305, -1.3460948, 1.3371561
9: -5.4464316, -3.8021157, -5.4913430, -3.7863193, -1.4801617, 1.5006444

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405203, upper bound: 0.6422874
time: 4.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422856, upper bound: 0.6422897
time: 4.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.5192966, -9.2470045, -11.4407139, -9.2574759, -1.6380389, 1.5879960
1: -6.5349269, -4.6733656, -6.4879580, -4.7400589, -1.6433864, 1.6198967
2: -6.2437210, -4.2171741, -6.1988630, -4.2688789, -1.5898924, 1.5990319
3: -5.3584542, -3.6847904, -5.3301487, -3.7690542, -1.2268798, 1.2268311
4: -7.4480247, -5.1529150, -7.3685179, -5.1723485, -1.6004024, 1.5703261
5: -10.4855251, -8.5823650, -10.4518709, -8.6196718, -1.3446465, 1.3440404
6: -17.1809673, -14.7020912, -17.1185684, -14.7238874, -1.6593332, 1.6461382
7: 5.0297513, 6.2556648, 5.0606399, 6.2385650, -1.0800455, 1.0736725
8: -6.4464092, -4.6263714, -6.4229445, -4.6886148, -1.3203626, 1.3371377
9: -5.4910398, -3.8014774, -5.4078889, -3.8463268, -1.4539897, 1.4244699

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6437974, upper bound: 0.6437954
time: 4.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6437949, upper bound: 0.6437950
time: 4.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.5212736, -9.2449646, -11.4713278, -9.2479782, -1.6465528, 1.6273141
1: -6.5370512, -4.6672096, -6.5234699, -4.7154393, -1.6588116, 1.6506915
2: -6.2503843, -4.2160387, -6.2258658, -4.2193756, -1.6256509, 1.6134772
3: -5.3648005, -3.6843164, -5.3554497, -3.7503898, -1.2527840, 1.2402427
4: -7.4495020, -5.1463542, -7.4019389, -5.1486120, -1.6039345, 1.6082997
5: -10.4868412, -8.5776634, -10.4817009, -8.6021147, -1.3518310, 1.3653619
6: -17.1838684, -14.7003565, -17.1340199, -14.7069950, -1.6718361, 1.6736287
7: 5.0275092, 6.2566943, 5.0498400, 6.2543097, -1.0937905, 1.0857916
8: -6.4502258, -4.6247816, -6.4465294, -4.6751719, -1.3408251, 1.3593628
9: -5.4927716, -3.7862957, -5.4514279, -3.7926896, -1.4614851, 1.4815884

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6437974, upper bound: 0.6454266
time: 4.29 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6437949, upper bound: 0.6454247
time: 7.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.4677734, -9.2534456, -11.4809484, -9.1961031, -1.6379774, 1.6199634
1: -6.5157404, -4.7188802, -6.5203676, -4.7120333, -1.6356387, 1.6324437
2: -6.2238855, -4.2203841, -6.2372913, -4.1535563, -1.6471908, 1.6295223
3: -5.3383713, -3.7518415, -5.3428459, -3.7393341, -1.2297318, 1.2233868
4: -7.3989377, -5.1582294, -7.4048805, -5.1412959, -1.6072993, 1.5947840
5: -10.4773235, -8.6045570, -10.4882488, -8.5606651, -1.3750386, 1.3649001
6: -17.1317863, -14.7180099, -17.1397114, -14.6926374, -1.6616650, 1.6513147
7: 5.0525694, 6.2496824, 5.0228024, 6.2554703, -1.0814195, 1.0941588
8: -6.4325776, -4.6761212, -6.4416952, -4.6380491, -1.3480682, 1.3278878
9: -5.4464316, -3.8021157, -5.4920688, -3.7938344, -1.4719632, 1.5054270

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6477622, upper bound: 0.6422892
time: 3.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6495278, upper bound: 0.6422893
time: 3.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.4677734, -9.2534456, -11.5339088, -9.1876249, -1.6482196, 1.6568604
1: -6.5157404, -4.7188802, -6.5415187, -4.6606579, -1.6602991, 1.6566660
2: -6.2238855, -4.2203841, -6.2637243, -4.1493359, -1.6517529, 1.6550680
3: -5.3383713, -3.7518415, -5.3692245, -3.6727502, -1.2386516, 1.2516204
4: -7.3989377, -5.1582294, -7.4543114, -5.1294127, -1.6206317, 1.6211287
5: -10.4773235, -8.6045570, -10.4977646, -8.5342770, -1.3829274, 1.3733730
6: -17.1317863, -14.7180099, -17.1902866, -14.6750164, -1.6810560, 1.6802191
7: 5.0525694, 6.2496824, 4.9982424, 6.2624812, -1.0892899, 1.1034218
8: -6.4325776, -4.6761212, -6.4593105, -4.5897627, -1.3558004, 1.3461807
9: -5.4464316, -3.8021157, -5.5369596, -3.7780521, -1.4884729, 1.5175997

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6477621, upper bound: 0.6422874
time: 5.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6495278, upper bound: 0.6422891
time: 3.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.5192966, -9.2470045, -11.4538956, -9.2001476, -1.6546321, 1.6014743
1: -6.5349269, -4.6733656, -6.4926229, -4.7332172, -1.6507106, 1.6240640
2: -6.2437210, -4.2171741, -6.2122655, -4.2020445, -1.6111121, 1.6120253
3: -5.3584542, -3.6847904, -5.3346176, -3.7565453, -1.2349569, 1.2316059
4: -7.4480247, -5.1529150, -7.3744621, -5.1554165, -1.6071062, 1.5754311
5: -10.4855251, -8.5823650, -10.4628000, -8.5757828, -1.3649013, 1.3549979
6: -17.1809673, -14.7020912, -17.1264935, -14.6984873, -1.6641042, 1.6543308
7: 5.0297513, 6.2556648, 5.0308867, 6.2443514, -1.0858593, 1.0922601
8: -6.4464092, -4.6263714, -6.4320464, -4.6505389, -1.3486652, 1.3461590
9: -5.4910398, -3.8014774, -5.4535265, -3.8380573, -1.4623098, 1.4577610

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6510360, upper bound: 0.6437965
time: 3.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6510360, upper bound: 0.6437943
time: 6.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.5212736, -9.2449646, -11.4844933, -9.1906376, -1.6631489, 1.6407677
1: -6.5370512, -4.6672096, -6.5280995, -4.7085953, -1.6614578, 1.6548951
2: -6.2503843, -4.2160387, -6.2392783, -4.1525488, -1.6468608, 1.6264594
3: -5.3648005, -3.6843164, -5.3599129, -3.7378819, -1.2577282, 1.2450194
4: -7.4495020, -5.1463542, -7.4078856, -5.1316805, -1.6106339, 1.6133990
5: -10.4868412, -8.5776634, -10.4926291, -8.5582247, -1.3702619, 1.3763186
6: -17.1838684, -14.7003565, -17.1419411, -14.6816101, -1.6765792, 1.6818209
7: 5.0275092, 6.2566943, 5.0200777, 6.2600961, -1.0996046, 1.1028341
8: -6.4502258, -4.6247816, -6.4556456, -4.6371002, -1.3630345, 1.3684094
9: -5.4927716, -3.7862957, -5.4970546, -3.7844050, -1.4697974, 1.4985406

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6510360, upper bound: 0.6454261
time: 3.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6510360, upper bound: 0.6454240
time: 5.12 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.4809484, -9.1961031, -11.4677734, -9.2534456, -1.6199636, 1.6379772
1: -6.5203676, -4.7120333, -6.5157404, -4.7188802, -1.6324439, 1.6356387
2: -6.2372913, -4.1535563, -6.2238855, -4.2203841, -1.6295223, 1.6471910
3: -5.3428459, -3.7393341, -5.3383713, -3.7518415, -1.2233865, 1.2297317
4: -7.4048805, -5.1412959, -7.3989377, -5.1582294, -1.5947838, 1.6072993
5: -10.4882488, -8.5606651, -10.4773235, -8.6045570, -1.3649001, 1.3750387
6: -17.1397114, -14.6926374, -17.1317863, -14.7180099, -1.6513147, 1.6616650
7: 5.0228024, 6.2554703, 5.0525694, 6.2496824, -1.0941589, 1.0814195
8: -6.4416952, -4.6380491, -6.4325776, -4.6761212, -1.3278880, 1.3480684
9: -5.4920688, -3.7938344, -5.4464316, -3.8021157, -1.5054271, 1.4719632

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405174, upper bound: 0.6495320
time: 3.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422850, upper bound: 0.6495320
time: 3.98 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.4809484, -9.1961031, -11.5207462, -9.2449646, -1.6301837, 1.6600136
1: -6.5203676, -4.7120333, -6.5369616, -4.6674967, -1.6618054, 1.6599123
2: -6.2372913, -4.1535563, -6.2503123, -4.2161674, -1.6347179, 1.6632857
3: -5.3428459, -3.7393341, -5.3647890, -3.6852601, -1.2397227, 1.2580101
4: -7.4048805, -5.1412959, -7.4483662, -5.1463513, -1.6081276, 1.6227388
5: -10.4882488, -8.5606651, -10.4868364, -8.5781603, -1.3856831, 1.3835269
6: -17.1397114, -14.6926374, -17.1823635, -14.7004156, -1.6705647, 1.6767714
7: 5.0228024, 6.2554703, 5.0279980, 6.2566948, -1.1020582, 1.0996742
8: -6.4416952, -4.6380491, -6.4501886, -4.6278305, -1.3551407, 1.3664745
9: -5.4920688, -3.7938344, -5.4913430, -3.7863193, -1.5219820, 1.5089747

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405174, upper bound: 0.6495298
time: 5.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422850, upper bound: 0.6495319
time: 3.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.5324631, -9.1896667, -11.4407139, -9.2574759, -1.6514854, 1.6172981
1: -6.5394936, -4.6665268, -6.4879580, -4.7400589, -1.6475816, 1.6225512
2: -6.2571249, -4.1503468, -6.1988630, -4.2688789, -1.6028800, 1.6256343
3: -5.3628931, -3.6722808, -5.3301487, -3.7690542, -1.2315896, 1.2305433
4: -7.4539766, -5.1359749, -7.3685179, -5.1723485, -1.6055024, 1.5871032
5: -10.4964542, -8.5384874, -10.4518709, -8.6196718, -1.3555946, 1.3522387
6: -17.1888905, -14.6766872, -17.1185684, -14.7238874, -1.6675324, 1.6649829
7: 5.0000010, 6.2614517, 5.0606399, 6.2385650, -1.0896194, 1.0794815
8: -6.4555283, -4.5883045, -6.4229445, -4.6886148, -1.3293910, 1.3468444
9: -5.5366554, -3.7932034, -5.4078889, -3.8463268, -1.4709604, 1.4327943

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6437968, upper bound: 0.6510377
time: 4.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6437944, upper bound: 0.6510362
time: 4.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.5344343, -9.1876240, -11.4713278, -9.2479782, -1.6599956, 1.6462567
1: -6.5416088, -4.6603699, -6.5234699, -4.7154393, -1.6629477, 1.6533511
2: -6.2637968, -4.1492085, -6.2258658, -4.2193756, -1.6386375, 1.6347429
3: -5.3692369, -3.6718066, -5.3554497, -3.7503898, -1.2574909, 1.2439547
4: -7.4554477, -5.1294141, -7.4019389, -5.1486120, -1.6090341, 1.6150107
5: -10.4977674, -8.5337811, -10.4817009, -8.6021147, -1.3627777, 1.3735628
6: -17.1917915, -14.6749544, -17.1340199, -14.7069950, -1.6800354, 1.6842151
7: 4.9977531, 6.2624807, 5.0498400, 6.2543097, -1.1033759, 1.0916008
8: -6.4593439, -4.5867138, -6.4465294, -4.6751719, -1.3498507, 1.3690684
9: -5.5383873, -3.7780271, -5.4514279, -3.7926896, -1.4784420, 1.4899070

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6437968, upper bound: 0.6526698
time: 4.11 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6437968, upper bound: 0.6526673
time: 6.29 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.4809484, -9.1961031, -11.4809484, -9.1961031, -1.6514194, 1.6514192
1: -6.5203676, -4.7120333, -6.5203676, -4.7120333, -1.6463096, 1.6463094
2: -6.2372913, -4.1535563, -6.2372913, -4.1535563, -1.6602037, 1.6602036
3: -5.3428459, -3.7393341, -5.3428459, -3.7393341, -1.2371187, 1.2371185
4: -7.4048805, -5.1412959, -7.4048805, -5.1412959, -1.6089635, 1.6089635
5: -10.4882488, -8.5606651, -10.4882488, -8.5606651, -1.3859944, 1.3859944
6: -17.1397114, -14.6926374, -17.1397114, -14.6926374, -1.6698642, 1.6698643
7: 5.0228024, 6.2554703, 5.0228024, 6.2554703, -1.0999725, 1.0999724
8: -6.4416952, -4.6380491, -6.4416952, -4.6380491, -1.3571141, 1.3571141
9: -5.4920688, -3.7938344, -5.4920688, -3.7938344, -1.5088522, 1.5088520

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405970, upper bound: 0.6495323
time: 3.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6423626, upper bound: 0.6495325
time: 4.14 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.4809484, -9.1961031, -11.5339088, -9.1876249, -1.6616616, 1.6734538
1: -6.5203676, -4.7120333, -6.5415187, -4.6606579, -1.6681001, 1.6705756
2: -6.2372913, -4.1535563, -6.2637243, -4.1493359, -1.6647654, 1.6762729
3: -5.3428459, -3.7393341, -5.3692245, -3.6727502, -1.2443973, 1.2653519
4: -7.4048805, -5.1412959, -7.4543114, -5.1294127, -1.6222961, 1.6283031
5: -10.4882488, -8.5606651, -10.4977646, -8.5342770, -1.3938832, 1.3944833
6: -17.1397114, -14.6926374, -17.1902866, -14.6750164, -1.6892557, 1.6849709
7: 5.0228024, 6.2554703, 4.9982424, 6.2624812, -1.1078722, 1.1092354
8: -6.4416952, -4.6380491, -6.4593105, -4.5897627, -1.3648465, 1.3755176
9: -5.4920688, -3.7938344, -5.5369596, -3.7780521, -1.5253625, 1.5259299

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405969, upper bound: 0.6495300
time: 4.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6423626, upper bound: 0.6495317
time: 4.22 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.5324631, -9.1896667, -11.4538956, -9.2001476, -1.6680779, 1.6307502
1: -6.5394936, -4.6665268, -6.4926229, -4.7332172, -1.6590557, 1.6303506
2: -6.2571249, -4.1503468, -6.2122655, -4.2020445, -1.6241002, 1.6386433
3: -5.3628931, -3.6722808, -5.3346176, -3.7565453, -1.2433981, 1.2362812
4: -7.4539766, -5.1359749, -7.3744621, -5.1554165, -1.6126735, 1.5896028
5: -10.4964542, -8.5384874, -10.4628000, -8.5757828, -1.3767021, 1.3631961
6: -17.1888905, -14.6766872, -17.1264935, -14.6984873, -1.6723034, 1.6731822
7: 5.0000010, 6.2614517, 5.0308867, 6.2443514, -1.0954332, 1.0980735
8: -6.4555283, -4.5883045, -6.4320464, -4.6505389, -1.3577082, 1.3558657
9: -5.5366554, -3.7932034, -5.4535265, -3.8380573, -1.4792805, 1.4660805

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6438710, upper bound: 0.6510379
time: 4.28 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6438710, upper bound: 0.6510386
time: 4.57 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.5344343, -9.1876240, -11.4844933, -9.1906376, -1.6765919, 1.6596932
1: -6.5416088, -4.6603699, -6.5280995, -4.7085953, -1.6692379, 1.6611865
2: -6.2637968, -4.1492085, -6.2392783, -4.1525488, -1.6598480, 1.6477492
3: -5.3692369, -3.6718066, -5.3599129, -3.7378819, -1.2634281, 1.2496946
4: -7.4554477, -5.1294141, -7.4078856, -5.1316805, -1.6162004, 1.6205759
5: -10.4977674, -8.5337811, -10.4926291, -8.5582247, -1.3812184, 1.3845193
6: -17.1917915, -14.6749544, -17.1419411, -14.6816101, -1.6847789, 1.6924149
7: 4.9977531, 6.2624807, 5.0200777, 6.2600961, -1.1091900, 1.1086483
8: -6.4593439, -4.5867138, -6.4556456, -4.6371002, -1.3720777, 1.3781151
9: -5.5383873, -3.7780271, -5.4970546, -3.7844050, -1.4867542, 1.5068595

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6438709, upper bound: 0.6526697
time: 3.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6438709, upper bound: 0.6526698
time: 4.38 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.00 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6405202, upper bound: 0.6422896
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6422856, upper bound: 0.6422898
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6405203, upper bound: 0.6422874
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6422856, upper bound: 0.6422897
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6437974, upper bound: 0.6437954
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6437949, upper bound: 0.6437950
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6437974, upper bound: 0.6454266
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6437949, upper bound: 0.6454247
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6477622, upper bound: 0.6422892
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6495278, upper bound: 0.6422893
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6477621, upper bound: 0.6422874
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6495278, upper bound: 0.6422891
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6510360, upper bound: 0.6437965
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6510360, upper bound: 0.6437943
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6510360, upper bound: 0.6454261
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6510360, upper bound: 0.6454240
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6405174, upper bound: 0.6495320
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6422850, upper bound: 0.6495320
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6405174, upper bound: 0.6495298
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6422850, upper bound: 0.6495319
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6437968, upper bound: 0.6510377
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6437944, upper bound: 0.6510362
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6437968, upper bound: 0.6526698
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6437968, upper bound: 0.6526673
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6405970, upper bound: 0.6495323
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6423626, upper bound: 0.6495325
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6405969, upper bound: 0.6495300
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6423626, upper bound: 0.6495317
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6438710, upper bound: 0.6510379
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6438710, upper bound: 0.6510386
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6438709, upper bound: 0.6526697
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.00
Output dim: 7, lower bound: -0.6438709, upper bound: 0.6526698

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4370708, -9.2629662, -11.4657745, -9.2554893, -1.5689216, 1.5957224
1: -6.4802380, -4.7434063, -6.5135965, -4.7252035, -1.5802960, 1.5995765
2: -6.1969080, -4.2698922, -6.2170682, -4.2215104, -1.5878778, 1.5638180
3: -5.3130426, -3.7705274, -5.3320189, -3.7523227, -1.1930394, 1.1927035
4: -7.3654976, -5.1819601, -7.3974700, -5.1647830, -1.5482302, 1.5641732
5: -10.4474611, -8.6221247, -10.4761038, -8.6092777, -1.3204808, 1.3330274
6: -17.1164589, -14.7349186, -17.1287308, -14.7197304, -1.6219587, 1.6182989
7: 5.0633945, 6.2338929, 5.0548081, 6.2487054, -1.0627611, 1.0554748
8: -6.4089937, -4.6895652, -6.4287772, -4.6777120, -1.2941418, 1.2984071
9: -5.4028192, -3.8558006, -5.4446740, -3.8173194, -1.3994777, 1.4067628

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405180, upper bound: 0.6405204
time: 3.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405180, upper bound: 0.6422902
time: 3.72 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 22.11 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.11
Output dim: 7, lower bound: -0.6405180, upper bound: 0.6405204
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.11
Output dim: 7, lower bound: -0.6405180, upper bound: 0.6422902
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6422856, upper bound: 0.6422898
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6405203, upper bound: 0.6422874
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6422856, upper bound: 0.6422897
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6437974, upper bound: 0.6437954
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6437949, upper bound: 0.6437950
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6437974, upper bound: 0.6454266
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6437949, upper bound: 0.6454247
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6477622, upper bound: 0.6422892
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6495278, upper bound: 0.6422893
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6477621, upper bound: 0.6422874
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6495278, upper bound: 0.6422891
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6510360, upper bound: 0.6437965
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6510360, upper bound: 0.6437943
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6510360, upper bound: 0.6454261
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6510360, upper bound: 0.6454240
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6405174, upper bound: 0.6495320
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6422850, upper bound: 0.6495320
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6405174, upper bound: 0.6495298
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6422850, upper bound: 0.6495319
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6437968, upper bound: 0.6510377
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6437944, upper bound: 0.6510362
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6437968, upper bound: 0.6526698
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6437968, upper bound: 0.6526673
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6405970, upper bound: 0.6495323
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6423626, upper bound: 0.6495325
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6405969, upper bound: 0.6495300
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6423626, upper bound: 0.6495317
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6438710, upper bound: 0.6510379
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6438710, upper bound: 0.6510386
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6438709, upper bound: 0.6526697
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.11
Output dim: 7, lower bound: -0.6438709, upper bound: 0.6526698
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.090501308441162
rel_dist={7: [-0.6526852532184364, 0.6526854163544584]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 577

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741247, upper bound: 0.4689571
time: 4.02 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741525, upper bound: 0.4741514
time: 4.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.69
Output dim: 7, lower bound: -0.4741247, upper bound: 0.4689571
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.69
Output dim: 7, lower bound: -0.4741525, upper bound: 0.4741514

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -11.4713383, -9.2479610, -11.4794312, -9.2467747, -1.3284740, 1.3366396
1: -6.5234876, -4.7154131, -6.5254960, -4.7152882, -1.4388382, 1.4400823
2: -6.2258978, -4.2193713, -6.2348447, -4.2183509, -1.4149041, 1.4239724
3: -5.3554783, -3.7503858, -5.3566294, -3.7477632, -1.0533583, 1.0535738
4: -7.4019489, -5.1485786, -7.4051113, -5.1483517, -1.3463838, 1.3488201
5: -10.4817095, -8.6020994, -10.4897232, -8.6006708, -1.1451063, 1.1529346
6: -17.1340351, -14.7069759, -17.1387138, -14.7062035, -1.3595867, 1.3641953
7: 5.0498300, 6.2543149, 5.0489011, 6.2585783, -0.9774961, 0.9733523
8: -6.4465570, -4.6751695, -6.4527354, -4.6739492, -1.1155405, 1.1217322
9: -5.4514394, -3.7926459, -5.4518294, -3.7870038, -1.3419042, 1.3364656

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689544, upper bound: 0.4689541
time: 3.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689544, upper bound: 0.4689543
time: 6.23 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.4845057, -9.1906204, -11.4819889, -9.2464142, -1.3412564, 1.3657589
1: -6.5281162, -4.7085695, -6.5261412, -4.7152481, -1.4515309, 1.4482496
2: -6.2393103, -4.1525426, -6.2376704, -4.2180395, -1.4273658, 1.4562328
3: -5.3599415, -3.7378778, -5.3569946, -3.7469797, -1.0585988, 1.0660733
4: -7.4078946, -5.1316462, -7.4061103, -5.1482806, -1.3529813, 1.3672926
5: -10.4926367, -8.5582094, -10.4922514, -8.6001902, -1.1555073, 1.1725892
6: -17.1419563, -14.6815882, -17.1401958, -14.7059727, -1.3672228, 1.3805681
7: 5.0200677, 6.2601023, 5.0486197, 6.2599235, -0.9959600, 0.9784303
8: -6.4556770, -4.6370978, -6.4546885, -4.6735840, -1.1236138, 1.1479380
9: -5.4970675, -3.7843614, -5.4519520, -3.7852209, -1.3779831, 1.3437040

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689544, upper bound: 0.4741244
time: 3.45 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689544, upper bound: 0.4741522
time: 3.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.86 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.86
Output dim: 7, lower bound: -0.4689544, upper bound: 0.4689541
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.86
Output dim: 7, lower bound: -0.4689544, upper bound: 0.4689543
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.86
Output dim: 7, lower bound: -0.4689544, upper bound: 0.4741244
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.86
Output dim: 7, lower bound: -0.4689544, upper bound: 0.4741522

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -11.4713383, -9.2479610, -11.4713383, -9.2479610, -1.3283942, 1.3283939
1: -6.5234876, -4.7154131, -6.5234876, -4.7154131, -1.4375324, 1.4375324
2: -6.2258978, -4.2193713, -6.2258978, -4.2193713, -1.4148655, 1.4148657
3: -5.3554783, -3.7503858, -5.3554783, -3.7503858, -1.0520635, 1.0520633
4: -7.4019489, -5.1485786, -7.4019489, -5.1485786, -1.3462474, 1.3462474
5: -10.4817095, -8.6020994, -10.4817095, -8.6020994, -1.1448984, 1.1448984
6: -17.1340351, -14.7069759, -17.1340351, -14.7069759, -1.3593817, 1.3593817
7: 5.0498300, 6.2543149, 5.0498300, 6.2543149, -0.9731205, 0.9731205
8: -6.4465570, -4.6751695, -6.4465570, -4.6751695, -1.1154804, 1.1154804
9: -5.4514394, -3.7926459, -5.4514394, -3.7926459, -1.3362403, 1.3362405

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689239, upper bound: 0.4667848
time: 5.70 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689526, upper bound: 0.4689517
time: 5.53 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -11.4713383, -9.2479610, -11.4845057, -9.1906204, -1.3548975, 1.3418467
1: -6.5234876, -4.7154131, -6.5281162, -4.7085695, -1.4448934, 1.4417043
2: -6.2258978, -4.2193713, -6.2393103, -4.1525426, -1.4442408, 1.4278693
3: -5.3554783, -3.7503858, -5.3599415, -3.7378778, -1.0631597, 1.0568089
4: -7.4019489, -5.1485786, -7.4078946, -5.1316462, -1.3638711, 1.3513538
5: -10.4817095, -8.6020994, -10.4926367, -8.5582094, -1.1620049, 1.1558440
6: -17.1340351, -14.7069759, -17.1419563, -14.6815882, -1.3742285, 1.3675742
7: 5.0498300, 6.2543149, 5.0200677, 6.2601023, -0.9789298, 0.9902005
8: -6.4465570, -4.6751695, -6.4556770, -4.6370978, -1.1397042, 1.1245091
9: -5.4514394, -3.7926459, -5.4970675, -3.7843614, -1.3445678, 1.3705243

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689239, upper bound: 0.4667848
time: 5.28 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689526, upper bound: 0.4689520
time: 6.01 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.4845057, -9.1906204, -11.4713383, -9.2479610, -1.3418467, 1.3548975
1: -6.5281162, -4.7085695, -6.5234876, -4.7154131, -1.4417043, 1.4448936
2: -6.2393103, -4.1525426, -6.2258978, -4.2193713, -1.4278688, 1.4442410
3: -5.3599415, -3.7378778, -5.3554783, -3.7503858, -1.0568089, 1.0631598
4: -7.4078946, -5.1316462, -7.4019489, -5.1485786, -1.3513539, 1.3638709
5: -10.4926367, -8.5582094, -10.4817095, -8.6020994, -1.1558437, 1.1620048
6: -17.1419563, -14.6815882, -17.1340351, -14.7069759, -1.3675742, 1.3742284
7: 5.0200677, 6.2601023, 5.0498300, 6.2543149, -0.9902005, 0.9789298
8: -6.4556770, -4.6370978, -6.4465570, -4.6751695, -1.1245091, 1.1397043
9: -5.4970675, -3.7843614, -5.4514394, -3.7926459, -1.3705244, 1.3445678

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689237, upper bound: 0.4719546
time: 6.75 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689524, upper bound: 0.4741215
time: 6.20 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.4845057, -9.1906204, -11.4845057, -9.1906204, -1.3676333, 1.3676333
1: -6.5281162, -4.7085695, -6.5281162, -4.7085695, -1.4539819, 1.4539824
2: -6.2393103, -4.1525426, -6.2393103, -4.1525426, -1.4567087, 1.4567089
3: -5.3599415, -3.7378778, -5.3599415, -3.7378778, -1.0693436, 1.0693436
4: -7.4078946, -5.1316462, -7.4078946, -5.1316462, -1.3645756, 1.3645759
5: -10.4926367, -8.5582094, -10.4926367, -8.5582094, -1.1723552, 1.1723553
6: -17.1419563, -14.6815882, -17.1419563, -14.6815882, -1.3818109, 1.3818110
7: 5.0200677, 6.2601023, 5.0200677, 6.2601023, -0.9952137, 0.9952137
8: -6.4556770, -4.6370978, -6.4556770, -4.6370978, -1.1477807, 1.1477805
9: -5.4970675, -3.7843614, -5.4970675, -3.7843614, -1.3776960, 1.3776958

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689237, upper bound: 0.4719819
time: 4.19 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689524, upper bound: 0.4689520
time: 6.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.70 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 7, lower bound: -0.4689239, upper bound: 0.4667848
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 7, lower bound: -0.4689526, upper bound: 0.4689517
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 7, lower bound: -0.4689239, upper bound: 0.4667848
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 7, lower bound: -0.4689526, upper bound: 0.4689520
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 7, lower bound: -0.4689237, upper bound: 0.4719546
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 7, lower bound: -0.4689524, upper bound: 0.4741215
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 7, lower bound: -0.4689237, upper bound: 0.4719819
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 7, lower bound: -0.4689524, upper bound: 0.4689520

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4677734, -9.2534456, -11.4708710, -9.2486897, -1.3224092, 1.3223765
1: -6.5157404, -4.7188802, -6.5224457, -4.7158594, -1.4273348, 1.4330845
2: -6.2238855, -4.2203841, -6.2256312, -4.2195101, -1.4131117, 1.4127426
3: -5.3383713, -3.7518415, -5.3532057, -3.7505639, -1.0351143, 1.0493264
4: -7.3989377, -5.1582294, -7.4015450, -5.1498594, -1.3400061, 1.3361129
5: -10.4773235, -8.6045570, -10.4811230, -8.6024113, -1.1397991, 1.1419454
6: -17.1317863, -14.7180099, -17.1337433, -14.7084446, -1.3551404, 1.3473870
7: 5.0525694, 6.2496824, 5.0501909, 6.2536988, -0.9694247, 0.9668185
8: -6.4325776, -4.6761212, -6.4446931, -4.6752839, -1.1014912, 1.1118867
9: -5.4464316, -3.8021157, -5.4508123, -3.7939060, -1.3297126, 1.3261087

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689217, upper bound: 0.4658963
time: 5.47 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689217, upper bound: 0.4667824
time: 6.10 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -11.5208616, -9.2449646, -11.4713326, -9.2479763, -1.3552196, 1.3340074
1: -6.5369825, -4.6674328, -6.5234704, -4.7154179, -1.4639783, 1.4616787
2: -6.2503300, -4.2161384, -6.2258935, -4.2193723, -1.4377105, 1.4220560
3: -5.3647928, -3.6850548, -5.3554626, -3.7503896, -1.0616612, 1.0659956
4: -7.4486141, -5.1463518, -7.4019432, -5.1485910, -1.3648484, 1.3488350
5: -10.4868393, -8.5780506, -10.4817047, -8.6021042, -1.1482720, 1.1594447
6: -17.1826935, -14.7004051, -17.1340294, -14.7070007, -1.3820519, 1.3662522
7: 5.0278912, 6.2566948, 5.0498352, 6.2543097, -0.9870858, 0.9750035
8: -6.4501987, -4.6271667, -6.4465361, -4.6751695, -1.1193664, 1.1363200
9: -5.4916549, -3.7863140, -5.4514313, -3.7926562, -1.3606398, 1.3424397

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689503, upper bound: 0.4680635
time: 4.88 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689504, upper bound: 0.4689497
time: 6.07 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.4677734, -9.2534456, -11.4840403, -9.1913471, -1.3489172, 1.3358314
1: -6.5157404, -4.7188802, -6.5270767, -4.7090149, -1.4346952, 1.4372609
2: -6.2238855, -4.2203841, -6.2390418, -4.1526823, -1.4425111, 1.4257445
3: -5.3383713, -3.7518415, -5.3576703, -3.7380567, -1.0462112, 1.0540727
4: -7.3989377, -5.1582294, -7.4074903, -5.1329269, -1.3576293, 1.3412186
5: -10.4773235, -8.6045570, -10.4920473, -8.5585194, -1.1568911, 1.1528912
6: -17.1317863, -14.7180099, -17.1416645, -14.6830597, -1.3697729, 1.3555793
7: 5.0525694, 6.2496824, 5.0204272, 6.2594862, -0.9752338, 0.9838892
8: -6.4325776, -4.6761212, -6.4538116, -4.6372104, -1.1256223, 1.1209149
9: -5.4464316, -3.8021157, -5.4964433, -3.7856207, -1.3380392, 1.3603733

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4740910, upper bound: 0.4658962
time: 4.28 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4740910, upper bound: 0.4667822
time: 6.65 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.5208616, -9.2449646, -11.4844990, -9.1906366, -1.3718154, 1.3474605
1: -6.5369825, -4.6674328, -6.5280995, -4.7085724, -1.4698830, 1.4656100
2: -6.2503300, -4.2161384, -6.2393060, -4.1525440, -1.4589143, 1.4350588
3: -5.3647928, -3.6850548, -5.3599262, -3.7378798, -1.0727577, 1.0704101
4: -7.4486141, -5.1463518, -7.4078894, -5.1316586, -1.3715541, 1.3539414
5: -10.4868393, -8.5780506, -10.4926310, -8.5582142, -1.1653862, 1.1673632
6: -17.1826935, -14.7004051, -17.1419544, -14.6816139, -1.3867962, 1.3744447
7: 5.0278912, 6.2566948, 5.0200725, 6.2600970, -0.9912794, 0.9921033
8: -6.4501987, -4.6271667, -6.4556541, -4.6371002, -1.1435947, 1.1430874
9: -5.4916549, -3.7863140, -5.4970598, -3.7843709, -1.3668461, 1.3767021

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741196, upper bound: 0.4680661
time: 3.72 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741197, upper bound: 0.4689494
time: 4.71 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.4809484, -9.1961031, -11.4708710, -9.2486897, -1.3358769, 1.3488859
1: -6.5203676, -4.7120333, -6.5224457, -4.7158594, -1.4315000, 1.4404445
2: -6.2372913, -4.1535563, -6.2256312, -4.2195101, -1.4261084, 1.4421138
3: -5.3428459, -3.7393341, -5.3532057, -3.7505639, -1.0398657, 1.0604225
4: -7.4048805, -5.1412959, -7.4015450, -5.1498594, -1.3451111, 1.3537329
5: -10.4882488, -8.5606651, -10.4811230, -8.6024113, -1.1507444, 1.1589503
6: -17.1397114, -14.6926374, -17.1337433, -14.7084446, -1.3633330, 1.3621638
7: 5.0228024, 6.2554703, 5.0501909, 6.2536988, -0.9863920, 0.9726275
8: -6.4416952, -4.6380491, -6.4446931, -4.6752839, -1.1105187, 1.1357599
9: -5.4920688, -3.7938344, -5.4508123, -3.7939060, -1.3637333, 1.3344331

Time for backsubstitution: 16.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689213, upper bound: 0.4710655
time: 5.61 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689213, upper bound: 0.4719517
time: 6.90 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.5340223, -9.1876259, -11.4713326, -9.2479763, -1.3656993, 1.3593723
1: -6.5415378, -4.6605954, -6.5234704, -4.7154179, -1.4681611, 1.4643383
2: -6.2637410, -4.1493068, -6.2258935, -4.2193723, -1.4474394, 1.4475576
3: -5.3692279, -3.6725442, -5.3554626, -3.7503896, -1.0663681, 1.0697068
4: -7.4545603, -5.1294117, -7.4019432, -5.1485910, -1.3695526, 1.3664441
5: -10.4977655, -8.5341692, -10.4817047, -8.6021042, -1.1592188, 1.1676455
6: -17.1906147, -14.6750040, -17.1340294, -14.7070007, -1.3886638, 1.3811649
7: 4.9981346, 6.2624812, 5.0498352, 6.2543097, -0.9966471, 0.9808128
8: -6.4593182, -4.5890989, -6.4465361, -4.6751695, -1.1283913, 1.1460258
9: -5.5372701, -3.7780461, -5.4514313, -3.7926562, -1.3775952, 1.3507509

Time for backsubstitution: 19.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689499, upper bound: 0.4732328
time: 3.96 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689500, upper bound: 0.4741191
time: 6.01 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.4809484, -9.1961031, -11.4840403, -9.1913471, -1.3616669, 1.3616239
1: -6.5203676, -4.7120333, -6.5270767, -4.7090149, -1.4437895, 1.4495456
2: -6.2372913, -4.1535563, -6.2390418, -4.1526823, -1.4549718, 1.4545801
3: -5.3428459, -3.7393341, -5.3576703, -3.7380567, -1.0524030, 1.0662483
4: -7.4048805, -5.1412959, -7.4074903, -5.1329269, -1.3583355, 1.3544384
5: -10.4882488, -8.5606651, -10.4920473, -8.5585194, -1.1672418, 1.1693009
6: -17.1397114, -14.6926374, -17.1416645, -14.6830597, -1.3773553, 1.3697462
7: 5.0228024, 6.2554703, 5.0204272, 6.2594862, -0.9914050, 0.9889019
8: -6.4416952, -4.6380491, -6.4538116, -4.6372104, -1.1336977, 1.1438358
9: -5.4920688, -3.7938344, -5.4964433, -3.7856207, -1.3709047, 1.3675417

Time for backsubstitution: 18.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690498, upper bound: 0.4710953
time: 4.09 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690499, upper bound: 0.4719801
time: 5.90 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.5340223, -9.1876259, -11.4844990, -9.1906366, -1.3845644, 1.3721083
1: -6.5415378, -4.6605954, -6.5280995, -4.7085724, -1.4767909, 1.4712815
2: -6.2637410, -4.1493068, -6.2393060, -4.1525440, -1.4713502, 1.4600252
3: -5.3692279, -3.6725442, -5.3599262, -3.7378798, -1.0790516, 1.0753061
4: -7.4545603, -5.1294117, -7.4078894, -5.1316586, -1.3765886, 1.3671494
5: -10.4977655, -8.5341692, -10.4926310, -8.5582142, -1.1757379, 1.1779958
6: -17.1906147, -14.6750040, -17.1419544, -14.6816139, -1.3943787, 1.3887473
7: 4.9981346, 6.2624812, 5.0200725, 6.2600970, -1.0016601, 0.9971161
8: -6.4593182, -4.5890989, -6.4556541, -4.6371002, -1.1516669, 1.1541017
9: -5.5372701, -3.7780461, -5.4970598, -3.7843709, -1.3847666, 1.3838584

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690784, upper bound: 0.4732611
time: 3.98 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690785, upper bound: 0.4741474
time: 4.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.28 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4689217, upper bound: 0.4658963
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4689217, upper bound: 0.4667824
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4689503, upper bound: 0.4680635
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4689504, upper bound: 0.4689497
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4740910, upper bound: 0.4658962
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4740910, upper bound: 0.4667822
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4741196, upper bound: 0.4680661
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4741197, upper bound: 0.4689494
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4689213, upper bound: 0.4710655
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4689213, upper bound: 0.4719517
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4689499, upper bound: 0.4732328
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4689500, upper bound: 0.4741191
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4690498, upper bound: 0.4710953
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4690499, upper bound: 0.4719801
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4690784, upper bound: 0.4732611
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.28
Output dim: 7, lower bound: -0.4690785, upper bound: 0.4741474

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.4643593, -9.2568874, -11.4402409, -9.2581978, -1.3101618, 1.2831368
1: -6.5120955, -4.7295222, -6.4869308, -4.7404881, -1.3967226, 1.3840346
2: -6.2124071, -4.2223210, -6.1986070, -4.2690177, -1.3553925, 1.3827336
3: -5.3276749, -3.7526779, -5.3278856, -3.7692318, -1.0049181, 1.0234524
4: -7.3964157, -5.1692772, -7.3681164, -5.1736212, -1.3132586, 1.2898202
5: -10.4752378, -8.6125145, -10.4512835, -8.6199808, -1.1176429, 1.1046886
6: -17.1266785, -14.7209187, -17.1182995, -14.7253437, -1.3294849, 1.3247216
7: 5.0563345, 6.2479935, 5.0610013, 6.2379475, -0.9476955, 0.9533173
8: -6.4262176, -4.6788397, -6.4210930, -4.6887283, -1.0783195, 1.0859299
9: -5.4434452, -3.8277533, -5.4072618, -3.8475823, -1.2711067, 1.2500310

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689207, upper bound: 0.4653043
time: 3.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689207, upper bound: 0.4658957
time: 4.49 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.4677696, -9.2534494, -11.4708652, -9.2486973, -1.3220859, 1.3225769
1: -6.5157380, -4.7188902, -6.5224400, -4.7158837, -1.4105780, 1.4296422
2: -6.2238736, -4.2203865, -6.2256031, -4.2195148, -1.4027977, 1.3937924
3: -5.3383636, -3.7518430, -5.3531866, -3.7505655, -1.0341527, 1.0355954
4: -7.3989353, -5.1582408, -7.4015412, -5.1498861, -1.3162479, 1.3298328
5: -10.4773216, -8.6045637, -10.4811172, -8.6024237, -1.1228638, 1.1362618
6: -17.1317806, -14.7180119, -17.1337318, -14.7084513, -1.3465309, 1.3500489
7: 5.0525737, 6.2496815, 5.0501986, 6.2536964, -0.9700792, 0.9657415
8: -6.4325705, -4.6761222, -6.4446769, -4.6752858, -1.1014838, 1.1108882
9: -5.4464293, -3.8021321, -5.4508071, -3.7939434, -1.2751272, 1.3115654

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689207, upper bound: 0.4661900
time: 5.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689207, upper bound: 0.4667819
time: 3.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.5174809, -9.2484055, -11.4407129, -9.2574835, -1.3427830, 1.2947717
1: -6.5333686, -4.6777968, -6.4879522, -4.7400603, -1.4333580, 1.4128790
2: -6.2391157, -4.2180929, -6.1988630, -4.2688808, -1.3800128, 1.3920336
3: -5.3541021, -3.6858799, -5.3301444, -3.7690563, -1.0314832, 1.0400171
4: -7.4460735, -5.1574173, -7.3685145, -5.1723547, -1.3381543, 1.3025085
5: -10.4845877, -8.5859823, -10.4518690, -8.6196737, -1.1260855, 1.1205282
6: -17.1778431, -14.7033348, -17.1185703, -14.7238979, -1.3559837, 1.3435667
7: 5.0316691, 6.2549176, 5.0606413, 6.2385626, -0.9653211, 0.9614114
8: -6.4438062, -4.6298838, -6.4229345, -4.6886158, -1.0961730, 1.1107097
9: -5.4887104, -3.8119209, -5.4078879, -3.8463306, -1.3018608, 1.2663918

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689494, upper bound: 0.4674700
time: 7.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689521, upper bound: 0.4680623
time: 4.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.5208588, -9.2449703, -11.4713240, -9.2479839, -1.3487937, 1.3330964
1: -6.5369787, -4.6674423, -6.5234642, -4.7154417, -1.4428477, 1.4448311
2: -6.2503171, -4.2161412, -6.2258639, -4.2193766, -1.4192007, 1.4002666
3: -5.3647847, -3.6850553, -5.3554430, -3.7503912, -1.0608490, 1.0506294
4: -7.4486089, -5.1463618, -7.4019384, -5.1486163, -1.3367639, 1.3425361
5: -10.4868364, -8.5780573, -10.4816990, -8.6021156, -1.1313365, 1.1449581
6: -17.1826859, -14.7004070, -17.1340179, -14.7070055, -1.3668230, 1.3688886
7: 5.0278940, 6.2566943, 5.0498424, 6.2543068, -0.9803607, 0.9739236
8: -6.4501896, -4.6271677, -6.4465189, -4.6751733, -1.1193585, 1.1316975
9: -5.4916515, -3.7863302, -5.4514256, -3.7926939, -1.3001704, 1.3279066

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689494, upper bound: 0.4683556
time: 6.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689495, upper bound: 0.4689487
time: 6.29 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.4643593, -9.2568874, -11.4534197, -9.2008677, -1.3368659, 1.2966161
1: -6.5120955, -4.7295222, -6.4915962, -4.7336454, -1.4040456, 1.3881993
2: -6.2124071, -4.2223210, -6.2120061, -4.2021818, -1.3846095, 1.3957253
3: -5.3276749, -3.7526779, -5.3323555, -3.7567239, -1.0136600, 1.0281956
4: -7.3964157, -5.1692772, -7.3740616, -5.1566868, -1.3308713, 1.2949247
5: -10.4752378, -8.6125145, -10.4622126, -8.5760899, -1.1347218, 1.1156366
6: -17.1266785, -14.7209187, -17.1262245, -14.6999454, -1.3440194, 1.3329142
7: 5.0563345, 6.2479935, 5.0312471, 6.2437344, -0.9535048, 0.9704080
8: -6.4262176, -4.6788397, -6.4301958, -4.6506515, -1.1015651, 1.0949302
9: -5.4434452, -3.8277533, -5.4528995, -3.8393118, -1.2794197, 1.2776681

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4740900, upper bound: 0.4653037
time: 5.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4740900, upper bound: 0.4658953
time: 3.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.4677696, -9.2534494, -11.4840317, -9.1913576, -1.3427081, 1.3330530
1: -6.5157380, -4.7188902, -6.5270696, -4.7090378, -1.4179323, 1.4335961
2: -6.2238736, -4.2203865, -6.2390132, -4.1526861, -1.4240088, 1.4045864
3: -5.3383636, -3.7518430, -5.3576517, -3.7380579, -1.0378652, 1.0403416
4: -7.3989353, -5.1582408, -7.4074850, -5.1329536, -1.3298295, 1.3345394
5: -10.4773216, -8.6045637, -10.4920425, -8.5585346, -1.1355224, 1.1441815
6: -17.1317806, -14.7180119, -17.1416550, -14.6830702, -1.3552055, 1.3582411
7: 5.0525737, 6.2496815, 5.0204353, 6.2594824, -0.9742723, 0.9787536
8: -6.4325705, -4.6761222, -6.4537950, -4.6372161, -1.1185210, 1.1199169
9: -5.4464293, -3.8021321, -5.4964371, -3.7856588, -1.2834206, 1.3285236

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4740900, upper bound: 0.4661892
time: 6.19 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4740900, upper bound: 0.4667813
time: 6.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.5174809, -9.2484055, -11.4538937, -9.2001524, -1.3593755, 1.3082492
1: -6.5333686, -4.6777968, -6.4926176, -4.7332191, -1.4392481, 1.4167984
2: -6.2391157, -4.2180929, -6.2122636, -4.2020440, -1.4012332, 1.4050267
3: -5.3541021, -3.6858799, -5.3346119, -3.7565460, -1.0403872, 1.0444274
4: -7.4460735, -5.1574173, -7.3744597, -5.1554213, -1.3448582, 1.3076131
5: -10.4845877, -8.5859823, -10.4627981, -8.5757828, -1.1431978, 1.1284490
6: -17.1778431, -14.7033348, -17.1264915, -14.6984978, -1.3607545, 1.3517590
7: 5.0316691, 6.2549176, 5.0308886, 6.2443504, -0.9695144, 0.9785297
8: -6.4438062, -4.6298838, -6.4320388, -4.6505380, -1.1195323, 1.1174519
9: -5.4887104, -3.8119209, -5.4535251, -3.8380606, -1.3080530, 1.2940456

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741187, upper bound: 0.4653028
time: 5.47 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741187, upper bound: 0.4680622
time: 4.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.5208588, -9.2449703, -11.4844952, -9.1906462, -1.3653891, 1.3435709
1: -6.5369787, -4.6674423, -6.5280943, -4.7085962, -1.4454947, 1.4487863
2: -6.2503171, -4.2161412, -6.2392759, -4.1525483, -1.4404116, 1.4100120
3: -5.3647847, -3.6850553, -5.3599072, -3.7378821, -1.0645614, 1.0550443
4: -7.4486089, -5.1463618, -7.4078836, -5.1316857, -1.3434634, 1.3472434
5: -10.4868364, -8.5780573, -10.4926252, -8.5582256, -1.1439407, 1.1528778
6: -17.1826859, -14.7004070, -17.1419430, -14.6816206, -1.3715670, 1.3770809
7: 5.0278940, 6.2566943, 5.0200801, 6.2600927, -0.9845541, 0.9869163
8: -6.4501896, -4.6271677, -6.4556394, -4.6371021, -1.1364930, 1.1384650
9: -5.4916515, -3.7863302, -5.4970527, -3.7844079, -1.3063555, 1.3448591

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741187, upper bound: 0.4683580
time: 4.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741188, upper bound: 0.4689485
time: 4.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.4775410, -9.1995506, -11.4402409, -9.2581978, -1.3236392, 1.3090179
1: -6.5167437, -4.7226753, -6.4869308, -4.7404881, -1.4009063, 1.3913863
2: -6.2258124, -4.1554956, -6.1986070, -4.2690177, -1.3683748, 1.4119487
3: -5.3321562, -3.7401681, -5.3278856, -3.7692318, -1.0096748, 1.0345497
4: -7.4023638, -5.1523428, -7.3681164, -5.1736212, -1.3183675, 1.3042748
5: -10.4861660, -8.5686293, -10.4512835, -8.6199808, -1.1285915, 1.1199889
6: -17.1346016, -14.6955452, -17.1182995, -14.7253437, -1.3376775, 1.3396790
7: 5.0265751, 6.2537808, 5.0610013, 6.2379475, -0.9646115, 0.9591269
8: -6.4353361, -4.6407666, -6.4210930, -4.6887283, -1.0873520, 1.1101629
9: -5.4890842, -3.8194594, -5.4072618, -3.8475823, -1.3050776, 1.2583780

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689231, upper bound: 0.4704735
time: 4.20 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689203, upper bound: 0.4710650
time: 4.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.4809446, -9.1961050, -11.4708652, -9.2486973, -1.3355527, 1.3391709
1: -6.5203648, -4.7120428, -6.5224400, -4.7158837, -1.4147422, 1.4322982
2: -6.2372780, -4.1535568, -6.2256031, -4.2195148, -1.4125490, 1.4160502
3: -5.3428383, -3.7393346, -5.3531866, -3.7505655, -1.0385725, 1.0453130
4: -7.4048786, -5.1413054, -7.4015412, -5.1498861, -1.3213520, 1.3365395
5: -10.4882469, -8.5606718, -10.4811172, -8.6024237, -1.1338091, 1.1444583
6: -17.1397057, -14.6926432, -17.1337318, -14.7084513, -1.3547239, 1.3566893
7: 5.0228043, 6.2554688, 5.0501986, 6.2536964, -0.9796662, 0.9715507
8: -6.4416895, -4.6380501, -6.4446769, -4.6752858, -1.1105113, 1.1311427
9: -5.4920673, -3.7938509, -5.4508071, -3.7939434, -1.3033333, 1.3177683

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689203, upper bound: 0.4713593
time: 5.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689203, upper bound: 0.4719514
time: 3.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.5306511, -9.1910667, -11.4407129, -9.2574835, -1.3532715, 1.3195210
1: -6.5379410, -4.6709585, -6.4879522, -4.7400603, -1.4375608, 1.4155350
2: -6.2525196, -4.1512690, -6.1988630, -4.2688808, -1.3897507, 1.4173722
3: -5.3585443, -3.6733704, -5.3301444, -3.7690563, -1.0361943, 1.0437303
4: -7.4520264, -5.1404772, -7.3685145, -5.1723547, -1.3428643, 1.3169754
5: -10.4955149, -8.5421028, -10.4518690, -8.6196737, -1.1370337, 1.1287259
6: -17.1857681, -14.6779308, -17.1185703, -14.7238979, -1.3625951, 1.3586260
7: 5.0019207, 6.2607040, 5.0606413, 6.2385626, -0.9749033, 0.9672201
8: -6.4529252, -4.5918179, -6.4229345, -4.6886158, -1.1052029, 1.1204163
9: -5.5343256, -3.8036435, -5.4078879, -3.8463306, -1.3188453, 1.2747242

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689517, upper bound: 0.4726390
time: 8.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689489, upper bound: 0.4732319
time: 4.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.5340176, -9.1876278, -11.4713240, -9.2479839, -1.3592780, 1.3496715
1: -6.5415359, -4.6606054, -6.5234642, -4.7154417, -1.4467483, 1.4474912
2: -6.2637277, -4.1493101, -6.2258639, -4.2193766, -1.4289286, 1.4214728
3: -5.3692198, -3.6725457, -5.3554430, -3.7503912, -1.0652249, 1.0543418
4: -7.4545574, -5.1294231, -7.4019384, -5.1486163, -1.3414738, 1.3492469
5: -10.4977627, -8.5341749, -10.4816990, -8.6021156, -1.1422832, 1.1531584
6: -17.1906109, -14.6750050, -17.1340179, -14.7070055, -1.3734343, 1.3756757
7: 4.9981384, 6.2624788, 5.0498424, 6.2543068, -0.9899457, 0.9797323
8: -6.4593120, -4.5890980, -6.4465189, -4.6751733, -1.1283836, 1.1414043
9: -5.5372672, -3.7780619, -5.4514256, -3.7926939, -1.3171306, 1.3340985

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689489, upper bound: 0.4735247
time: 5.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689490, upper bound: 0.4741182
time: 4.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.4775410, -9.1995506, -11.4534197, -9.2008677, -1.3496249, 1.3217797
1: -6.5167437, -4.7226753, -6.4915962, -4.7336454, -1.4131486, 1.4004793
2: -6.2258124, -4.1554956, -6.2120061, -4.2021818, -1.3970568, 1.4244043
3: -5.3321562, -3.7401681, -5.3323555, -3.7567239, -1.0192637, 1.0403236
4: -7.4023638, -5.1523428, -7.3740616, -5.1566868, -1.3315835, 1.3081375
5: -10.4861660, -8.5686293, -10.4622126, -8.5760899, -1.1450748, 1.1303416
6: -17.1346016, -14.6955452, -17.1262245, -14.6999454, -1.3516016, 1.3472613
7: 5.0265751, 6.2537808, 5.0312471, 6.2437344, -0.9696245, 0.9754207
8: -6.4353361, -4.6407666, -6.4301958, -4.6506515, -1.1096425, 1.1182134
9: -5.4890842, -3.8194594, -5.4528995, -3.8393118, -1.3122346, 1.2848431

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690488, upper bound: 0.4705016
time: 4.95 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690488, upper bound: 0.4710948
time: 3.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.4809446, -9.1961050, -11.4840317, -9.1913576, -1.3554626, 1.3519166
1: -6.5203648, -4.7120428, -6.5270696, -4.7090378, -1.4262524, 1.4392650
2: -6.2372780, -4.1535568, -6.2390132, -4.1526861, -1.4364686, 1.4285033
3: -5.3428383, -3.7393346, -5.3576517, -3.7380579, -1.0434704, 1.0509130
4: -7.4048786, -5.1413054, -7.4074850, -5.1329536, -1.3345685, 1.3415756
5: -10.4882469, -8.5606718, -10.4920425, -8.5585346, -1.1458740, 1.1548096
6: -17.1397057, -14.6926432, -17.1416550, -14.6830702, -1.3627882, 1.3642720
7: 5.0228043, 6.2554688, 5.0204353, 6.2594824, -0.9846790, 0.9837661
8: -6.4416895, -4.6380501, -6.4537950, -4.6372161, -1.1265962, 1.1392187
9: -5.4920673, -3.7938509, -5.4964371, -3.7856588, -1.3104837, 1.3356923

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690488, upper bound: 0.4713871
time: 5.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690488, upper bound: 0.4719785
time: 4.12 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.5306511, -9.1910667, -11.4538937, -9.2001524, -1.3721335, 1.3322809
1: -6.5379410, -4.6709585, -6.4926176, -4.7332191, -1.4461722, 1.4224665
2: -6.2525196, -4.1512690, -6.2122636, -4.2020440, -1.4136786, 1.4298296
3: -5.3585443, -3.6733704, -5.3346119, -3.7565460, -1.0459452, 1.0493273
4: -7.4520264, -5.1404772, -7.3744597, -5.1554213, -1.3498991, 1.3208272
5: -10.4955149, -8.5421028, -10.4627981, -8.5757828, -1.1535509, 1.1390784
6: -17.1857681, -14.6779308, -17.1264915, -14.6984978, -1.3683372, 1.3662088
7: 5.0019207, 6.2607040, 5.0308886, 6.2443504, -0.9799161, 0.9835423
8: -6.4529252, -4.5918179, -6.4320388, -4.6505380, -1.1276069, 1.1284672
9: -5.5343256, -3.8036435, -5.4535251, -3.8380606, -1.3260031, 1.3012073

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690802, upper bound: 0.4726693
time: 4.29 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690775, upper bound: 0.4732603
time: 4.24 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.5340176, -9.1876278, -11.4844952, -9.1906462, -1.3781424, 1.3624161
1: -6.5415359, -4.6606054, -6.5280943, -4.7085962, -1.4524150, 1.4544580
2: -6.2637277, -4.1493101, -6.2392759, -4.1525483, -1.4528465, 1.4339265
3: -5.3692198, -3.6725457, -5.3599072, -3.7378821, -1.0701206, 1.0599415
4: -7.4545574, -5.1294231, -7.4078836, -5.1316857, -1.3485038, 1.3542836
5: -10.4977627, -8.5341749, -10.4926252, -8.5582256, -1.1542926, 1.1635098
6: -17.1906109, -14.6750050, -17.1419430, -14.6816206, -1.3791494, 1.3832582
7: 4.9981384, 6.2624788, 5.0200801, 6.2600927, -0.9949586, 0.9919294
8: -6.4593120, -4.5890980, -6.4556394, -4.6371021, -1.1445656, 1.1494803
9: -5.5372672, -3.7780619, -5.4970527, -3.7844079, -1.3242807, 1.3520155

Time for backsubstitution: 14.56 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=0.9791851043701172
rel_dist={7: [-0.4741558180834273, 0.47415573392575094]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 577

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111670, upper bound: 0.4073550
time: 6.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111799, upper bound: 0.4111762
time: 4.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.37 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.37
Output dim: 7, lower bound: -0.4111670, upper bound: 0.4073550
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.37
Output dim: 7, lower bound: -0.4111799, upper bound: 0.4111762

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -11.4713383, -9.2479610, -11.4776888, -9.2470217, -1.2322180, 1.2386285
1: -6.5234876, -4.7154131, -6.5250602, -4.7153149, -1.3706651, 1.3716412
2: -6.2258978, -4.2193713, -6.2329226, -4.2185655, -1.3465338, 1.3536541
3: -5.3554783, -3.7503858, -5.3563814, -3.7483091, -0.9916999, 0.9919329
4: -7.4019489, -5.1485786, -7.4044313, -5.1484013, -1.2603869, 1.2622695
5: -10.4817095, -8.6020994, -10.4880018, -8.6009693, -1.0730042, 1.0791507
6: -17.1340351, -14.7069759, -17.1377068, -14.7063656, -1.2601805, 1.2637979
7: 5.0498300, 6.2543149, 5.0490980, 6.2576623, -0.9394505, 0.9361972
8: -6.4465570, -4.6751695, -6.4514065, -4.6742048, -1.0425537, 1.0474148
9: -5.4514394, -3.7926459, -5.4517469, -3.7882156, -1.2933536, 1.2890828

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4073556, upper bound: 0.4073552
time: 4.38 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4073584, upper bound: 0.4073583
time: 3.99 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.4845057, -9.1906204, -11.4819870, -9.2464142, -1.2446008, 1.2678990
1: -6.5281162, -4.7085695, -6.5261402, -4.7152481, -1.3831160, 1.3803573
2: -6.2393103, -4.1525426, -6.2376685, -4.2180405, -1.3586721, 1.3874534
3: -5.3599415, -3.7378778, -5.3569946, -3.7469802, -0.9972825, 1.0039973
4: -7.4078946, -5.1316462, -7.4061103, -5.1482801, -1.2666945, 1.2807378
5: -10.4926367, -8.5582094, -10.4922514, -8.6001902, -1.0830858, 1.0992011
6: -17.1419563, -14.6815882, -17.1401939, -14.7059727, -1.2674899, 1.2799518
7: 5.0200677, 6.2601023, 5.0486202, 6.2599235, -0.9583645, 0.9408418
8: -6.4556770, -4.6370978, -6.4546843, -4.6735830, -1.0500555, 1.0732733
9: -5.4970675, -3.7843614, -5.4519515, -3.7852216, -1.3281317, 1.2956681

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073556, upper bound: 0.4111638
time: 4.54 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073556, upper bound: 0.4111768
time: 4.69 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.69 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 23.69
Output dim: 7, lower bound: -0.4073556, upper bound: 0.4073552
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 23.69
Output dim: 7, lower bound: -0.4073584, upper bound: 0.4073583
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.69
Output dim: 7, lower bound: -0.4073556, upper bound: 0.4111638
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.69
Output dim: 7, lower bound: -0.4073556, upper bound: 0.4111768

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.4845037, -9.1909552, -11.4713383, -9.2479610, -1.2456062, 1.2567332
1: -6.5280848, -4.7085686, -6.5234876, -4.7154131, -1.3737264, 1.3770022
2: -6.2393112, -4.1529417, -6.2258978, -4.2193713, -1.3595080, 1.3751138
3: -5.3599424, -3.7379687, -5.3554783, -3.7503858, -0.9954925, 1.0015935
4: -7.4078932, -5.1316457, -7.4019489, -5.1485786, -1.2653325, 1.2773876
5: -10.4926348, -8.5586205, -10.4817095, -8.6020994, -1.0837865, 1.0882218
6: -17.1419563, -14.6817207, -17.1340351, -14.7069759, -1.2682123, 1.2734752
7: 5.0201721, 6.2601023, 5.0498300, 6.2543149, -0.9524977, 0.9418242
8: -6.4556761, -4.6372709, -6.4465570, -4.6751695, -1.0515347, 1.0648602
9: -5.4968791, -3.7843614, -5.4514394, -3.7926459, -1.3205137, 1.2972329

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4072681, upper bound: 0.4095319
time: 4.85 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073538, upper bound: 0.4111644
time: 3.90 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.4845057, -9.1906204, -11.4845057, -9.1906204, -1.2693553, 1.2693553
1: -6.5281162, -4.7085695, -6.5281162, -4.7085695, -1.3855672, 1.3855672
2: -6.2393103, -4.1525426, -6.2393103, -4.1525426, -1.3875961, 1.3875961
3: -5.3599415, -3.7378778, -5.3599415, -3.7378778, -1.0072680, 1.0072680
4: -7.4078946, -5.1316462, -7.4078946, -5.1316462, -1.2782886, 1.2782887
5: -10.4926367, -8.5582094, -10.4926367, -8.5582094, -1.0986021, 1.0986021
6: -17.1419563, -14.6815882, -17.1419563, -14.6815882, -1.2808220, 1.2808222
7: 5.0200677, 6.2601023, 5.0200677, 6.2601023, -0.9571345, 0.9571345
8: -6.4556770, -4.6370978, -6.4556770, -4.6370978, -1.0725296, 1.0725297
9: -5.4970675, -3.7843614, -5.4970675, -3.7843614, -1.3271427, 1.3271422

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4072681, upper bound: 0.4095499
time: 4.10 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073565, upper bound: 0.4111649
time: 4.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.57 seconds
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.57
Output dim: 7, lower bound: -0.4072681, upper bound: 0.4095319
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.57
Output dim: 7, lower bound: -0.4073538, upper bound: 0.4111644
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.57
Output dim: 7, lower bound: -0.4072681, upper bound: 0.4095499
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.57
Output dim: 7, lower bound: -0.4073565, upper bound: 0.4111649

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.4809465, -9.1964340, -11.4702063, -9.2497196, -1.2386355, 1.2497735
1: -6.5203357, -4.7120323, -6.5209789, -4.7164993, -1.3629365, 1.3707132
2: -6.2372904, -4.1539545, -6.2252531, -4.2197075, -1.3573847, 1.3727169
3: -5.3428459, -3.7394247, -5.3499899, -3.7508223, -0.9784527, 0.9950269
4: -7.4048791, -5.1412950, -7.4009762, -5.1516733, -1.2573071, 1.2663777
5: -10.4882498, -8.5610771, -10.4802904, -8.6028585, -1.0782614, 1.0841801
6: -17.1397114, -14.6927710, -17.1333313, -14.7105236, -1.2617700, 1.2609278
7: 5.0229063, 6.2554703, 5.0507054, 6.2528286, -0.9474783, 0.9349754
8: -6.4416943, -4.6382208, -6.4420562, -4.6754532, -1.0372398, 1.0580249
9: -5.4918814, -3.7938347, -5.4499092, -3.7956870, -1.3117819, 1.2860281

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4072663, upper bound: 0.4088664
time: 4.06 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4072663, upper bound: 0.4095301
time: 4.23 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.5338659, -9.1879568, -11.4713297, -9.2479782, -1.2660003, 1.2602900
1: -6.5414791, -4.6606798, -6.5234690, -4.7154188, -1.3986819, 1.3957112
2: -6.2637186, -4.1497464, -6.2258935, -4.2193727, -1.3766484, 1.3782285
3: -5.3692231, -3.6729136, -5.3554611, -3.7503901, -1.0026572, 1.0073450
4: -7.4542236, -5.1294112, -7.4019408, -5.1485939, -1.2822251, 1.2783253
5: -10.4977646, -8.5347290, -10.4817028, -8.6021051, -1.0867848, 1.0937709
6: -17.1901703, -14.6751528, -17.1340313, -14.7070055, -1.2868845, 1.2794211
7: 4.9983840, 6.2624812, 5.0498357, 6.2543092, -0.9588275, 0.9436190
8: -6.4593072, -4.5901718, -6.4465322, -4.6751714, -1.0545695, 1.0707140
9: -5.5366597, -3.7780561, -5.4514294, -3.7926569, -1.3272839, 1.3020408

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073521, upper bound: 0.4104952
time: 4.21 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073520, upper bound: 0.4111592
time: 4.05 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.4809484, -9.1961031, -11.4833755, -9.1923790, -1.2622881, 1.2623999
1: -6.5203676, -4.7120333, -6.5256090, -4.7096539, -1.3747904, 1.3792906
2: -6.2372913, -4.1535563, -6.2386637, -4.1528788, -1.3854780, 1.3851967
3: -5.3428459, -3.7393341, -5.3544574, -3.7383156, -0.9902328, 1.0007031
4: -7.4048805, -5.1412959, -7.4069209, -5.1347423, -1.2702649, 1.2672311
5: -10.4882488, -8.5606651, -10.4912186, -8.5589676, -1.0930626, 1.0945610
6: -17.1397114, -14.6926374, -17.1412506, -14.6851444, -1.2739844, 1.2682754
7: 5.0228024, 6.2554703, 5.0209408, 6.2586164, -0.9521151, 0.9502776
8: -6.4416952, -4.6380491, -6.4511757, -4.6373792, -1.0581431, 1.0656941
9: -5.4920688, -3.7938344, -5.4955406, -3.7874041, -1.3184094, 1.3158939

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4074209, upper bound: 0.4088812
time: 4.81 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4074210, upper bound: 0.4095449
time: 4.43 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.5338669, -9.1876259, -11.4844971, -9.1906385, -1.2862375, 1.2729125
1: -6.5415125, -4.6606798, -6.5280986, -4.7085733, -1.4061992, 1.4021299
2: -6.2637186, -4.1493459, -6.2393050, -4.1525445, -1.4021957, 1.3907114
3: -5.3692236, -3.6728237, -5.3599248, -3.7378809, -1.0145776, 1.0130194
4: -7.4542246, -5.1294117, -7.4078865, -5.1316600, -1.2892303, 1.2791867
5: -10.4977646, -8.5343170, -10.4926300, -8.5582132, -1.1016061, 1.1041517
6: -17.1901703, -14.6750202, -17.1419544, -14.6816158, -1.2931862, 1.2867686
7: 4.9982781, 6.2624812, 5.0200720, 6.2600956, -0.9634640, 0.9589481
8: -6.4593058, -4.5899992, -6.4556494, -4.6370993, -1.0755668, 1.0783836
9: -5.5368490, -3.7780552, -5.4970584, -3.7843726, -1.3339124, 1.3319285

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075067, upper bound: 0.4105087
time: 4.28 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075066, upper bound: 0.4111727
time: 4.11 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.05 seconds
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.05
Output dim: 7, lower bound: -0.4072663, upper bound: 0.4088664
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 7, lower bound: -0.4072663, upper bound: 0.4095301
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 7, lower bound: -0.4073521, upper bound: 0.4104952
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 7, lower bound: -0.4073520, upper bound: 0.4111592
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.05
Output dim: 7, lower bound: -0.4074209, upper bound: 0.4088812
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 7, lower bound: -0.4074210, upper bound: 0.4095449
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 7, lower bound: -0.4075067, upper bound: 0.4105087
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.05
Output dim: 7, lower bound: -0.4075066, upper bound: 0.4111727

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.4809446, -9.1964397, -11.4701986, -9.2497282, -1.2345424, 1.2399523
1: -6.5203319, -4.7120447, -6.5209732, -4.7165227, -1.3444364, 1.3617573
2: -6.2372737, -4.1539574, -6.2252240, -4.2197123, -1.3414195, 1.3445567
3: -5.3428359, -3.7394269, -5.3499708, -3.7508249, -0.9762650, 0.9786711
4: -7.4048767, -5.1413093, -7.4009719, -5.1517005, -1.2322543, 1.2487074
5: -10.4882479, -8.5610847, -10.4802856, -8.6028719, -1.0579464, 1.0696795
6: -17.1397038, -14.6927748, -17.1333160, -14.7105303, -1.2531142, 1.2541953
7: 5.0229092, 6.2554679, 5.0507126, 6.2528257, -0.9406357, 0.9333018
8: -6.4416857, -4.6382213, -6.4420395, -4.6754532, -1.0372307, 1.0525972
9: -5.4918780, -3.7938559, -5.4499025, -3.7957249, -1.2477374, 1.2655396

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4072656, upper bound: 0.4090821
time: 4.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4072656, upper bound: 0.4095295
time: 4.09 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.5298605, -9.1920242, -11.4407110, -9.2574854, -1.2528770, 1.2199209
1: -6.5372162, -4.6729150, -6.4879494, -4.7400608, -1.3672469, 1.3463837
2: -6.2504749, -4.1520853, -6.1988635, -4.2688813, -1.3185244, 1.3474300
3: -5.3566060, -3.6739025, -5.3301425, -3.7690544, -0.9705842, 0.9812369
4: -7.4512053, -5.1424870, -7.3685145, -5.1723552, -1.2550004, 1.2268460
5: -10.4950876, -8.5441036, -10.4518681, -8.6196747, -1.0640216, 1.0537626
6: -17.1844673, -14.6786156, -17.1185684, -14.7239027, -1.2603772, 1.2561953
7: 5.0028548, 6.2603569, 5.0606427, 6.2385621, -0.9363527, 0.9297140
8: -6.4517784, -4.5934048, -6.4229288, -4.6886153, -1.0301781, 1.0445597
9: -5.5331702, -3.8083050, -5.4078870, -3.8463323, -1.2677388, 1.2206676

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073514, upper bound: 0.4100267
time: 4.31 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073514, upper bound: 0.4104946
time: 4.33 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.5338650, -9.1879616, -11.4713211, -9.2479858, -1.2582297, 1.2505059
1: -6.5414762, -4.6606932, -6.5234618, -4.7154436, -1.3745236, 1.3788631
2: -6.2637033, -4.1497488, -6.2258654, -4.2193766, -1.3581376, 1.3500304
3: -5.3692117, -3.6729143, -5.3554401, -3.7503910, -1.0006169, 0.9909527
4: -7.4542198, -5.1294255, -7.4019365, -5.1486192, -1.2521136, 1.2606586
5: -10.4977617, -8.5347338, -10.4816990, -8.6021156, -1.0663950, 1.0792841
6: -17.1901646, -14.6751566, -17.1340218, -14.7070112, -1.2716553, 1.2726884
7: 4.9983873, 6.2624793, 5.0498428, 6.2543044, -0.9520140, 0.9419067
8: -6.4592981, -4.5901718, -6.4465165, -4.6751738, -1.0545607, 1.0652815
9: -5.5366573, -3.7780764, -5.4514236, -3.7926958, -1.2631669, 1.2815841

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073514, upper bound: 0.4106902
time: 3.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073513, upper bound: 0.4111587
time: 4.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.4809456, -9.1961060, -11.4833679, -9.1923866, -1.2547808, 1.2525874
1: -6.5203633, -4.7120457, -6.5256033, -4.7096767, -1.3546882, 1.3681972
2: -6.2372746, -4.1535568, -6.2386341, -4.1528835, -1.3669755, 1.3570235
3: -5.3428354, -3.7393360, -5.3544393, -3.7383161, -0.9812968, 0.9843483
4: -7.4048772, -5.1413088, -7.4069166, -5.1347661, -1.2436628, 1.2534236
5: -10.4882469, -8.5606737, -10.4912148, -8.5589809, -1.0700374, 1.0800614
6: -17.1397038, -14.6926422, -17.1412373, -14.6851492, -1.2594180, 1.2615423
7: 5.0228062, 6.2554679, 5.0209489, 6.2586117, -0.9452722, 0.9442024
8: -6.4416862, -4.6380506, -6.4511576, -4.6373811, -1.0510397, 1.0602666
9: -5.4920659, -3.7938552, -5.4955359, -3.7874413, -1.2543440, 1.2840383

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4074202, upper bound: 0.4091017
time: 4.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4074228, upper bound: 0.4095441
time: 4.96 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.5298576, -9.1916904, -11.4538918, -9.2001572, -1.2731116, 1.2325671
1: -6.5372486, -4.6729150, -6.4926152, -4.7332191, -1.3747578, 1.3527911
2: -6.2504745, -4.1516848, -6.2122631, -4.2020445, -1.3440878, 1.3599023
3: -5.3566065, -3.6738133, -5.3346095, -3.7565453, -0.9799010, 0.9869088
4: -7.4512072, -5.1424870, -7.3744593, -5.1554241, -1.2620041, 1.2306786
5: -10.4950876, -8.5436916, -10.4627981, -8.5757856, -1.0788467, 1.0641448
6: -17.1844673, -14.6784830, -17.1264935, -14.6984987, -1.2667050, 1.2635427
7: 5.0027514, 6.2603579, 5.0308895, 6.2443500, -0.9409895, 0.9450622
8: -6.4517779, -4.5932322, -6.4320326, -4.6505384, -1.0504484, 1.0522034
9: -5.5333576, -3.8083034, -5.4535236, -3.8380632, -1.2743535, 1.2457578

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075086, upper bound: 0.4100458
time: 4.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075060, upper bound: 0.4105081
time: 4.27 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.5338631, -9.1876278, -11.4844933, -9.1906481, -1.2784669, 1.2631358
1: -6.5415087, -4.6606932, -6.5280919, -4.7085972, -1.3799014, 1.3853058
2: -6.2637033, -4.1493473, -6.2392759, -4.1525488, -1.3836911, 1.3624988
3: -5.3692136, -3.6728263, -5.3599048, -3.7378819, -1.0056460, 0.9966271
4: -7.4542208, -5.1294270, -7.4078808, -5.1316853, -1.2591128, 1.2653755
5: -10.4977617, -8.5343237, -10.4926262, -8.5582266, -1.0784860, 1.0896653
6: -17.1901646, -14.6750259, -17.1419411, -14.6816263, -1.2779567, 1.2800353
7: 4.9982839, 6.2624793, 5.0200801, 6.2600932, -0.9566500, 0.9528054
8: -6.4592986, -4.5899992, -6.4556341, -4.6371040, -1.0684638, 1.0729508
9: -5.5368452, -3.7780752, -5.4970527, -3.7844107, -1.2697744, 1.3000847

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075061, upper bound: 0.4107095
time: 4.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4075086, upper bound: 0.4111711
time: 5.05 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.26 seconds
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4072656, upper bound: 0.4090821
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4072656, upper bound: 0.4095295
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4073514, upper bound: 0.4100267
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4073514, upper bound: 0.4104946
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4073514, upper bound: 0.4106902
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4073513, upper bound: 0.4111587
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4074202, upper bound: 0.4091017
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4074228, upper bound: 0.4095441
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4075086, upper bound: 0.4100458
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4075060, upper bound: 0.4105081
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4075061, upper bound: 0.4107095
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.26
Output dim: 7, lower bound: -0.4075086, upper bound: 0.4111711

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.4809437, -9.1964397, -11.4701977, -9.2497282, -1.2282410, 1.2277982
1: -6.5203271, -4.7120490, -6.5209689, -4.7165232, -1.3444252, 1.3576787
2: -6.2372637, -4.1539598, -6.2252178, -4.2197137, -1.3295989, 1.3326461
3: -5.3428230, -3.7394295, -5.3499641, -3.7508266, -0.9724376, 0.9741969
4: -7.4048610, -5.1413107, -7.4009614, -5.1517024, -1.2121708, 1.2302445
5: -10.4882450, -8.5611067, -10.4802866, -8.6028843, -1.0329912, 1.0342432
6: -17.1396923, -14.6927767, -17.1333122, -14.7105341, -1.2363863, 1.2343075
7: 5.0229144, 6.2554669, 5.0507154, 6.2528238, -0.9320028, 0.9251091
8: -6.4416823, -4.6382227, -6.4420385, -4.6754541, -1.0179639, 1.0439497
9: -5.4918756, -3.7938714, -5.4499011, -3.7957339, -1.2331586, 1.2425381

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057210, upper bound: 0.4095295
time: 4.17 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057210, upper bound: 0.4095327
time: 3.76 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.5213909, -9.2012043, -11.4393892, -9.2586174, -1.2347078, 1.2052050
1: -6.5170507, -4.6837769, -6.4824018, -4.7426929, -1.3418722, 1.3261508
2: -6.2312908, -4.1725917, -6.1893902, -4.2698979, -1.2977135, 1.3182123
3: -5.3399334, -3.6874917, -5.3224945, -3.7719932, -0.9525023, 0.9608362
4: -7.4231586, -5.1595836, -7.3546276, -5.1730866, -1.2272999, 1.1925776
5: -10.4589615, -8.5795021, -10.4484100, -8.6392651, -1.0079134, 1.0135248
6: -17.1663074, -14.6983643, -17.1084557, -14.7261486, -1.2404702, 1.2236553
7: 5.0149870, 6.2535443, 5.0636644, 6.2374525, -0.9219973, 0.9152887
8: -6.4400525, -4.6012239, -6.4211354, -4.6909356, -1.0084498, 1.0254235
9: -5.5102282, -3.8283734, -5.4059672, -3.8572366, -1.2293310, 1.1977437

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057237, upper bound: 0.4099294
time: 3.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057237, upper bound: 0.4100269
time: 6.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.5298586, -9.1920252, -11.4407129, -9.2574854, -1.2464793, 1.2079914
1: -6.5372114, -4.6729188, -6.4879460, -4.7400632, -1.3662133, 1.3426758
2: -6.2504625, -4.1520863, -6.1988559, -4.2688828, -1.3081713, 1.3355194
3: -5.3565936, -3.6739042, -5.3301339, -3.7690558, -0.9702446, 0.9766810
4: -7.4511900, -5.1424890, -7.3685064, -5.1723580, -1.2307663, 1.2083832
5: -10.4950848, -8.5441170, -10.4518681, -8.6196852, -1.0415990, 1.0186172
6: -17.1844559, -14.6786156, -17.1185627, -14.7239037, -1.2441287, 1.2363091
7: 5.0028596, 6.2603569, 5.0606451, 6.2385612, -0.9279962, 0.9258473
8: -6.4517756, -4.5934048, -6.4229298, -4.6886158, -1.0138538, 1.0371149
9: -5.5331678, -3.8083186, -5.4078856, -3.8463407, -1.2530022, 1.2018926

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057237, upper bound: 0.4104092
time: 6.68 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057210, upper bound: 0.4104981
time: 3.97 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.5254097, -9.1970100, -11.4699993, -9.2492361, -1.2401481, 1.2358553
1: -6.5214229, -4.6716213, -6.5180197, -4.7181473, -1.3499112, 1.3589087
2: -6.2444978, -4.1702700, -6.2163525, -4.2204313, -1.3375757, 1.3219625
3: -5.3525229, -3.6865661, -5.3477573, -3.7533967, -0.9823904, 0.9706941
4: -7.4258938, -5.1464615, -7.3883028, -5.1493683, -1.2243659, 1.2272340
5: -10.4616518, -8.5701275, -10.4782372, -8.6217098, -1.0077050, 1.0393988
6: -17.1716995, -14.6949139, -17.1238194, -14.7092562, -1.2516682, 1.2403605
7: 5.0103636, 6.2556133, 5.0529432, 6.2531242, -0.9376473, 0.9275584
8: -6.4480915, -4.5980024, -6.4440470, -4.6775198, -1.0331247, 1.0455037
9: -5.5135560, -3.7981722, -5.4493828, -3.8036563, -1.2250342, 1.2584080

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057237, upper bound: 0.4105922
time: 6.17 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057210, upper bound: 0.4106899
time: 7.05 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.5338593, -9.1879625, -11.4713240, -9.2479887, -1.2517109, 1.2383649
1: -6.5414715, -4.6606960, -6.5234585, -4.7154441, -1.3715262, 1.3748713
2: -6.2636924, -4.1497493, -6.2258573, -4.2193775, -1.3462329, 1.3381199
3: -5.3691993, -3.6729167, -5.3554330, -3.7503920, -0.9967912, 0.9864907
4: -7.4542046, -5.1294270, -7.4019270, -5.1486201, -1.2277093, 1.2421962
5: -10.4977589, -8.5347567, -10.4816971, -8.6021299, -1.0413659, 1.0438523
6: -17.1901512, -14.6751575, -17.1340122, -14.7070141, -1.2551091, 1.2528017
7: 4.9983931, 6.2624779, 5.0498447, 6.2543049, -0.9433839, 0.9336438
8: -6.4592943, -4.5901747, -6.4465127, -4.6751733, -1.0353878, 1.0566198
9: -5.5366549, -3.7780924, -5.4514222, -3.7927053, -1.2485161, 1.2584863

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057210, upper bound: 0.4110729
time: 4.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057237, upper bound: 0.4111618
time: 4.28 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.4809418, -9.1961069, -11.4833689, -9.1923885, -1.2484920, 1.2404361
1: -6.5203590, -4.7120485, -6.5255985, -4.7096796, -1.3515499, 1.3641419
2: -6.2372642, -4.1535592, -6.2386279, -4.1528850, -1.3551633, 1.3451122
3: -5.3428221, -3.7393382, -5.3544312, -3.7383189, -0.9774709, 0.9798739
4: -7.4048610, -5.1413102, -7.4069080, -5.1347675, -1.2191746, 1.2349612
5: -10.4882460, -8.5606937, -10.4912109, -8.5589933, -1.0450823, 1.0446250
6: -17.1396923, -14.6926470, -17.1412334, -14.6851501, -1.2427139, 1.2416550
7: 5.0228100, 6.2554669, 5.0209503, 6.2586117, -0.9366388, 0.9360259
8: -6.4416828, -4.6380510, -6.4511566, -4.6373816, -1.0297692, 1.0516289
9: -5.4920640, -3.7938704, -5.4955335, -3.7874513, -1.2397649, 1.2610393

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058762, upper bound: 0.4095440
time: 5.07 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058762, upper bound: 0.4095474
time: 3.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.5213890, -9.2008724, -11.4525709, -9.2012844, -1.2549508, 1.2178559
1: -6.5170813, -4.6837769, -6.4871001, -4.7358522, -1.3495593, 1.3325843
2: -6.2312903, -4.1721916, -6.2027912, -4.2030663, -1.3232889, 1.3306689
3: -5.3399343, -3.6874013, -5.3269663, -3.7594838, -0.9618244, 0.9665083
4: -7.4231586, -5.1595831, -7.3605709, -5.1561527, -1.2343071, 1.1964142
5: -10.4589615, -8.5790901, -10.4593391, -8.5953789, -1.0199995, 1.0239086
6: -17.1663074, -14.6982298, -17.1163807, -14.7007475, -1.2468131, 1.2310021
7: 5.0148830, 6.2535453, 5.0339088, 6.2432394, -0.9266338, 0.9306076
8: -6.4400525, -4.6010513, -6.4302473, -4.6528583, -1.0284333, 1.0330794
9: -5.5104165, -3.8283713, -5.4516053, -3.8489528, -1.2359548, 1.2227762

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058788, upper bound: 0.4099481
time: 5.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058762, upper bound: 0.4100466
time: 4.28 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.5298576, -9.1916924, -11.4538908, -9.2001553, -1.2667272, 1.2206399
1: -6.5372434, -4.6729188, -6.4926119, -4.7332220, -1.3715942, 1.3491089
2: -6.2504635, -4.1516867, -6.2122564, -4.2020450, -1.3337429, 1.3479916
3: -5.3565950, -3.6738138, -5.3346024, -3.7565477, -0.9763546, 0.9823532
4: -7.4511909, -5.1424870, -7.3744526, -5.1554232, -1.2377753, 1.2130922
5: -10.4950848, -8.5437050, -10.4627962, -8.5757952, -1.0536857, 1.0290000
6: -17.1844559, -14.6784859, -17.1264877, -14.6985025, -1.2504816, 1.2436558
7: 5.0027542, 6.2603569, 5.0308909, 6.2443485, -0.9326327, 0.9367535
8: -6.4517775, -4.5932331, -6.4320316, -4.6505404, -1.0292590, 1.0447705
9: -5.5333562, -3.8083169, -5.4535227, -3.8380706, -1.2596171, 1.2231095

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058761, upper bound: 0.4104221
time: 4.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058761, upper bound: 0.4105115
time: 3.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.5254078, -9.1966791, -11.4831696, -9.1918983, -1.2603931, 1.2484903
1: -6.5214529, -4.6716204, -6.5226841, -4.7113023, -1.3552890, 1.3653272
2: -6.2444978, -4.1698704, -6.2297668, -4.1536078, -1.3631375, 1.3344164
3: -5.3525229, -3.6864772, -5.3522258, -3.7408888, -0.9874220, 0.9763697
4: -7.4258943, -5.1464610, -7.3942471, -5.1324358, -1.2313688, 1.2319458
5: -10.4616518, -8.5697145, -10.4891672, -8.5778208, -1.0197954, 1.0497839
6: -17.1716995, -14.6947823, -17.1317444, -14.6838722, -1.2579849, 1.2477075
7: 5.0102587, 6.2556124, 5.0231781, 6.2589111, -0.9422843, 0.9384775
8: -6.4480934, -4.5978298, -6.4531727, -4.6394510, -1.0471940, 1.0531851
9: -5.5137453, -3.7981715, -5.4950142, -3.7953596, -1.2316499, 1.2769229

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058762, upper bound: 0.4106125
time: 5.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4058762, upper bound: 0.4068744
time: 5.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.5338602, -9.1876307, -11.4844913, -9.1906500, -1.2719598, 1.2509983
1: -6.5415030, -4.6606956, -6.5280876, -4.7085991, -1.3769052, 1.3813375
2: -6.2636924, -4.1493483, -6.2392702, -4.1525497, -1.3717945, 1.3505883
3: -5.3691998, -3.6728265, -5.3598971, -3.7378836, -1.0018207, 0.9921649
4: -7.4542050, -5.1294270, -7.4078727, -5.1316872, -1.2347126, 1.2469134
5: -10.4977589, -8.5343447, -10.4926252, -8.5582399, -1.0534570, 1.0542345
6: -17.1901512, -14.6750259, -17.1419334, -14.6816282, -1.2614346, 1.2601488
7: 4.9982877, 6.2624779, 5.0200820, 6.2600918, -0.9480212, 0.9445598
8: -6.4592957, -4.5900021, -6.4556336, -4.6371031, -1.0471931, 1.0642997
9: -5.5368443, -3.7780919, -5.4970508, -3.7844193, -1.2551234, 1.2769903

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058788, upper bound: 0.4110861
time: 5.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058760, upper bound: 0.4111754
time: 3.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 24.08 seconds
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4057210, upper bound: 0.4095295
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4057210, upper bound: 0.4095327
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4057237, upper bound: 0.4099294
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4057237, upper bound: 0.4100269
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4057237, upper bound: 0.4104092
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4057210, upper bound: 0.4104981
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4057237, upper bound: 0.4105922
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4057210, upper bound: 0.4106899
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4057210, upper bound: 0.4110729
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4057237, upper bound: 0.4111618
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4058762, upper bound: 0.4095440
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4058762, upper bound: 0.4095474
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4058788, upper bound: 0.4099481
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4058762, upper bound: 0.4100466
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4058761, upper bound: 0.4104221
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4058761, upper bound: 0.4105115
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4058762, upper bound: 0.4106125
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4058762, upper bound: 0.4068744
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4058788, upper bound: 0.4110861
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 24.08
Output dim: 7, lower bound: -0.4058760, upper bound: 0.4111754

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.4809437, -9.1964397, -11.4677639, -9.2534542, -1.2248631, 1.2241507
1: -6.5203271, -4.7120490, -6.5157299, -4.7189064, -1.3424344, 1.3514807
2: -6.2372637, -4.1539598, -6.2238498, -4.2203894, -1.3283336, 1.3317133
3: -5.3428230, -3.7394295, -5.3383441, -3.7518454, -0.9721023, 0.9634001
4: -7.4048610, -5.1413107, -7.3989229, -5.1582546, -1.2061918, 1.2268305
5: -10.4882450, -8.5611067, -10.4773178, -8.6045837, -1.0314062, 1.0311154
6: -17.1396923, -14.6927767, -17.1317692, -14.7180176, -1.2288201, 1.2323859
7: 5.0229144, 6.2554669, 5.0525799, 6.2496791, -0.9281700, 0.9231865
8: -6.4416823, -4.6382227, -6.4325600, -4.6761250, -1.0167203, 1.0351714
9: -5.4918756, -3.7938714, -5.4464240, -3.8021638, -1.2272263, 1.2390662

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050592, upper bound: 0.4095291
time: 5.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4050592, upper bound: 0.4088655
time: 6.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.4809437, -9.1964397, -11.5200119, -9.2449760, -1.2337198, 1.2454250
1: -6.5203271, -4.7120490, -6.5368338, -4.6679230, -1.3579159, 1.3741475
2: -6.2372637, -4.1539598, -6.2501740, -4.2163482, -1.3327513, 1.3478463
3: -5.3428230, -3.7394295, -5.3647447, -3.6865668, -0.9773704, 0.9869022
4: -7.4048610, -5.1413107, -7.4467850, -5.1463814, -1.2165985, 1.2382237
5: -10.4882450, -8.5611067, -10.4868269, -8.5788727, -1.0389605, 1.0381122
6: -17.1396923, -14.6927767, -17.1802635, -14.7005072, -1.2454047, 1.2463032
7: 5.0229144, 6.2554669, 5.0286856, 6.2566905, -0.9343607, 0.9320117
8: -6.4416823, -4.6382227, -6.4501200, -4.6320457, -1.0222564, 1.0489887
9: -5.4918756, -3.7938714, -5.4893603, -3.7864065, -1.2407274, 1.2496794

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050592, upper bound: 0.4095295
time: 7.88 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4050564, upper bound: 0.4088652
time: 6.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.5207033, -9.2012043, -11.4357185, -9.2640896, -1.2291834, 1.1997472
1: -6.5169449, -4.6841545, -6.4746943, -4.7460465, -1.3283305, 1.3161920
2: -6.2311893, -4.1727552, -6.1874456, -4.2709017, -1.2956185, 1.3162895
3: -5.3399181, -3.6887217, -5.3053656, -3.7734673, -0.9555817, 0.9430060
4: -7.4216785, -5.1595840, -7.3516026, -5.1826944, -1.2173431, 1.1874577
5: -10.4589577, -8.5801516, -10.4440079, -8.6417284, -1.0049598, 1.0082828
6: -17.1643410, -14.6984367, -17.1063652, -14.7371655, -1.2278337, 1.2199005
7: 5.0156236, 6.2535443, 5.0663881, 6.2327776, -0.9155149, 0.9125650
8: -6.4400077, -4.6051998, -6.4072037, -4.6918869, -1.0081344, 1.0095630
9: -5.5083675, -3.8284097, -5.4008799, -3.8667231, -1.2186532, 1.1947534

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050592, upper bound: 0.4099290
time: 6.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050564, upper bound: 0.4099297
time: 6.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.5223141, -9.2012014, -11.4893951, -9.2555628, -1.2387116, 1.2215240
1: -6.5172024, -4.6832685, -6.4963732, -4.6935301, -1.3556871, 1.3411258
2: -6.2314267, -4.1723661, -6.2143979, -4.2665091, -1.3012199, 1.3336568
3: -5.3399601, -3.6858294, -5.3318729, -3.7053080, -0.9637589, 0.9695750
4: -7.4251518, -5.1595802, -7.4028378, -5.1708145, -1.2253089, 1.2003757
5: -10.4589653, -8.5786276, -10.4531832, -8.6144447, -1.0140781, 1.0168442
6: -17.1689510, -14.6982603, -17.1601677, -14.7194567, -1.2477248, 1.2373148
7: 5.0141287, 6.2535443, 5.0409822, 6.2395539, -0.9242215, 0.9268706
8: -6.4401145, -4.5958834, -6.4248643, -4.6384993, -1.0252843, 1.0313909
9: -5.5127306, -3.8283167, -5.4481044, -3.8507900, -1.2360415, 1.2123758

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050565, upper bound: 0.4100266
time: 5.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050565, upper bound: 0.4100268
time: 5.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.5291748, -9.1920271, -11.4370708, -9.2629642, -1.2409873, 1.2025234
1: -6.5371037, -4.6732960, -6.4802341, -4.7434092, -1.3535719, 1.3327082
2: -6.2503614, -4.1522541, -6.1969018, -4.2698932, -1.3060806, 1.3335562
3: -5.3565788, -3.6751359, -5.3130360, -3.7705281, -0.9701059, 0.9588915
4: -7.4497137, -5.1424899, -7.3654909, -5.1819596, -1.2208009, 1.2033079
5: -10.4950800, -8.5447655, -10.4474611, -8.6221342, -1.0386639, 1.0133247
6: -17.1824894, -14.6786928, -17.1164551, -14.7349195, -1.2315238, 1.2324398
7: 5.0035005, 6.2603569, 5.0633965, 6.2338915, -0.9215043, 0.9215951
8: -6.4517298, -4.5973825, -6.4089909, -4.6895661, -1.0125771, 1.0212679
9: -5.5313058, -3.8083560, -5.4028177, -3.8558090, -1.2423501, 1.1984458

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050565, upper bound: 0.4104088
time: 6.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050565, upper bound: 0.4104094
time: 7.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.5307827, -9.1920242, -11.4907198, -9.2545090, -1.2505209, 1.2243412
1: -6.5373645, -4.6724100, -6.5017481, -4.6909747, -1.3776326, 1.3578914
2: -6.2505989, -4.1518593, -6.2238159, -4.2654800, -1.3116813, 1.3509260
3: -5.3566189, -3.6722498, -5.3395128, -3.7023902, -0.9782763, 0.9854324
4: -7.4531832, -5.1424837, -7.4166842, -5.1700640, -1.2325976, 1.2160988
5: -10.4950914, -8.5432444, -10.4567232, -8.5948420, -1.0477459, 1.0218165
6: -17.1870956, -14.6785145, -17.1701069, -14.7172565, -1.2514575, 1.2496690
7: 5.0019946, 6.2603569, 5.0379686, 6.2407150, -0.9302318, 0.9330561
8: -6.4518394, -4.5880661, -6.4266534, -4.6361814, -1.0261109, 1.0431101
9: -5.5356684, -3.8082669, -5.4500303, -3.8399134, -1.2597003, 1.2127209

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050565, upper bound: 0.4104952
time: 3.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4050565, upper bound: 0.4104955
time: 5.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.5247250, -9.1970119, -11.4664145, -9.2546988, -1.2346201, 1.2303827
1: -6.5213099, -4.6719990, -6.5102983, -4.7216225, -1.3418798, 1.3489269
2: -6.2443976, -4.1704350, -6.2143512, -4.2214375, -1.3354690, 1.3199902
3: -5.3525057, -3.6877975, -5.3306408, -3.7548506, -0.9811696, 0.9528553
4: -7.4244127, -5.1464639, -7.3852921, -5.1590085, -1.2143869, 1.2221203
5: -10.4616470, -8.5707741, -10.4738617, -8.6241741, -1.0047503, 1.0341358
6: -17.1697350, -14.6949921, -17.1215820, -14.7202606, -1.2390375, 1.2365392
7: 5.0109940, 6.2556133, 5.0556502, 6.2484951, -0.9311881, 0.9232781
8: -6.4480462, -4.6019783, -6.4300961, -4.6784720, -1.0305090, 1.0295776
9: -5.5116944, -3.7982104, -5.4443693, -3.8131318, -1.2143383, 1.2523580

Time for backsubstitution: 14.26 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=0.9420795440673828
rel_dist={7: [-0.4111819478955816, 0.4111785782606754]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2418.72 seconds
