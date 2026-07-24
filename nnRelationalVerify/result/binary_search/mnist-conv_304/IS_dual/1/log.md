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
execution time: IAR + LP analysis = 15.49 + 32.75 = 48.24 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.76 seconds, max iter: 100)

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
rel_dist={7: [-0.3441224975867083, 0.3441223455793496]}

## Binary Search Result
Binary search time: 201.43 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3350.33 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 577

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526786, upper bound: 0.6454348
time: 4.41 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526785, upper bound: 0.6526784
time: 3.93 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.54 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.54
Output dim: 7, lower bound: -0.6526786, upper bound: 0.6454348
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.54
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

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6454344
time: 4.12 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6454342
time: 3.58 seconds

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

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6526787
time: 3.82 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6526787
time: 3.93 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.59 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.59
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6454344
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.59
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6454342
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.59
Output dim: 7, lower bound: -0.6454344, upper bound: 0.6526787
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.59
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

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454336, upper bound: 0.6445134
time: 4.02 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454335, upper bound: 0.6454331
time: 3.81 seconds

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

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454336, upper bound: 0.6445137
time: 3.75 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454360, upper bound: 0.6454325
time: 4.54 seconds

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

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445132, upper bound: 0.6526763
time: 5.74 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454330, upper bound: 0.6526772
time: 4.05 seconds

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

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454331, upper bound: 0.6517433
time: 6.93 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454331, upper bound: 0.6526796
time: 4.21 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.88 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.6454336, upper bound: 0.6445134
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.6454335, upper bound: 0.6454331
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.6454336, upper bound: 0.6445137
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.6454360, upper bound: 0.6454325
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.6445132, upper bound: 0.6526763
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.6454330, upper bound: 0.6526772
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.6454331, upper bound: 0.6517433
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.6454331, upper bound: 0.6526796

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4630165, -9.2571163, -11.4707031, -9.2485571, -1.6001146, 1.6031783
1: -6.5030842, -4.7265863, -6.5209069, -4.7167268, -1.6180420, 1.6248052
2: -6.2067380, -4.2399020, -6.2212577, -4.2198739, -1.6002073, 1.5930610
3: -5.3389597, -3.7640829, -5.3517675, -3.7517982, -1.2184250, 1.2207476
4: -7.3735528, -5.1655774, -7.3953495, -5.1489377, -1.5766840, 1.5745370
5: -10.4457083, -8.6375017, -10.4800892, -8.6115541, -1.3179903, 1.3226682
6: -17.1152649, -14.7268171, -17.1290894, -14.7080526, -1.6383805, 1.6253664
7: 5.0619850, 6.2475471, 5.0513110, 6.2537551, -1.0705140, 1.0719029
8: -6.4353361, -4.6829929, -6.4453793, -4.6762657, -1.3131857, 1.3159776
9: -5.4283996, -3.8126690, -5.4504690, -3.7979376, -1.4477444, 1.4565701

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422902, upper bound: 0.6445075
time: 6.03 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454272, upper bound: 0.6445078
time: 4.28 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -11.4713373, -9.2479630, -11.4713373, -9.2479610, -1.6210790, 1.6111696
1: -6.5234799, -4.7154164, -6.5234861, -4.7154131, -1.6412010, 1.6394129
2: -6.2258859, -4.2193723, -6.2258945, -4.2193699, -1.6189499, 1.6166975
3: -5.3554649, -3.7503891, -5.3554745, -3.7503874, -1.2358854, 1.2371771
4: -7.4019337, -5.1485777, -7.4019461, -5.1485791, -1.5930200, 1.5998509
5: -10.4817057, -8.6021194, -10.4817085, -8.6021051, -1.3490379, 1.3391054
6: -17.1340256, -14.7069778, -17.1340332, -14.7069740, -1.6584282, 1.6495910
7: 5.0498333, 6.2543139, 5.0498319, 6.2543144, -1.0828722, 1.0838810
8: -6.4465537, -4.6751709, -6.4465556, -4.6751704, -1.3221371, 1.3440902
9: -5.4514370, -3.7926612, -5.4514380, -3.7926483, -1.4782372, 1.4638026

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454274, upper bound: 0.6422903
time: 4.56 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454274, upper bound: 0.6454275
time: 4.59 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.4630165, -9.2571163, -11.4838705, -9.1912203, -1.6314745, 1.6166327
1: -6.5030842, -4.7265863, -6.5255523, -4.7098818, -1.6254034, 1.6289477
2: -6.2067380, -4.2399020, -6.2346706, -4.1530495, -1.6309893, 1.6060438
3: -5.3389597, -3.7640829, -5.3562336, -3.7392907, -1.2295215, 1.2254963
4: -7.3735528, -5.1655774, -7.4012957, -5.1320043, -1.5943074, 1.5796444
5: -10.4457083, -8.6375017, -10.4910154, -8.5676641, -1.3328476, 1.3336143
6: -17.1152649, -14.7268171, -17.1370125, -14.6826668, -1.6570330, 1.6335595
7: 5.0619850, 6.2475471, 5.0215483, 6.2595429, -1.0763230, 1.0904305
8: -6.4353361, -4.6829929, -6.4545035, -4.6381946, -1.3425431, 1.3250113
9: -5.4283996, -3.8126690, -5.4961004, -3.7896469, -1.4560826, 1.4983931

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6495325, upper bound: 0.6445070
time: 6.26 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526704, upper bound: 0.6445072
time: 4.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.4713373, -9.2479630, -11.4845047, -9.1906223, -1.6448834, 1.6246226
1: -6.5234799, -4.7154164, -6.5281162, -4.7085686, -1.6485617, 1.6435843
2: -6.2258859, -4.2193723, -6.2393064, -4.1525431, -1.6434350, 1.6297007
3: -5.3554649, -3.7503891, -5.3599386, -3.7378786, -1.2469821, 1.2419231
4: -7.4019337, -5.1485777, -7.4078903, -5.1316462, -1.6077914, 1.6049418
5: -10.4817057, -8.6021194, -10.4926348, -8.5582123, -1.3572345, 1.3500512
6: -17.1340256, -14.7069778, -17.1419563, -14.6815891, -1.6670990, 1.6577835
7: 5.0498333, 6.2543139, 5.0200686, 6.2601018, -1.0886817, 1.0965527
8: -6.4465537, -4.6751709, -6.4556761, -4.6370997, -1.3478832, 1.3531289
9: -5.4514370, -3.7926612, -5.4970665, -3.7843659, -1.4865642, 1.5028428

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526706, upper bound: 0.6422900
time: 4.53 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526706, upper bound: 0.6454269
time: 4.28 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -11.4838705, -9.1912203, -11.4630165, -9.2571163, -1.6166329, 1.6314744
1: -6.5255523, -4.7098818, -6.5030842, -4.7265863, -1.6289477, 1.6254032
2: -6.2346706, -4.1530495, -6.2067380, -4.2399020, -1.6060433, 1.6309893
3: -5.3562336, -3.7392907, -5.3389597, -3.7640829, -1.2254961, 1.2295215
4: -7.4012957, -5.1320043, -7.3735528, -5.1655774, -1.5796444, 1.5943073
5: -10.4910154, -8.5676641, -10.4457083, -8.6375017, -1.3336146, 1.3328476
6: -17.1370125, -14.6826668, -17.1152649, -14.7268171, -1.6335597, 1.6570330
7: 5.0215483, 6.2595429, 5.0619850, 6.2475471, -1.0904305, 1.0763230
8: -6.4545035, -4.6381946, -6.4353361, -4.6829929, -1.3250110, 1.3425432
9: -5.4961004, -3.7896469, -5.4283996, -3.8126690, -1.4983928, 1.4560826

Time for backsubstitution: 12.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445069, upper bound: 0.6495325
time: 7.04 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445069, upper bound: 0.6526703
time: 6.67 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -11.4845047, -9.1906223, -11.4713373, -9.2479630, -1.6246226, 1.6448836
1: -6.5281162, -4.7085686, -6.5234799, -4.7154164, -1.6435843, 1.6485617
2: -6.2393064, -4.1525431, -6.2258859, -4.2193723, -1.6297002, 1.6434351
3: -5.3599386, -3.7378786, -5.3554649, -3.7503891, -1.2419231, 1.2469819
4: -7.4078903, -5.1316462, -7.4019337, -5.1485777, -1.6049418, 1.6077914
5: -10.4926348, -8.5582123, -10.4817057, -8.6021194, -1.3500509, 1.3572344
6: -17.1419563, -14.6815891, -17.1340256, -14.7069778, -1.6577835, 1.6670989
7: 5.0200686, 6.2601018, 5.0498333, 6.2543139, -1.0965528, 1.0886817
8: -6.4556761, -4.6370997, -6.4465537, -4.6751709, -1.3531289, 1.3478832
9: -5.4970665, -3.7843659, -5.4514370, -3.7926612, -1.5028429, 1.4865642

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422895, upper bound: 0.6526706
time: 5.62 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454266, upper bound: 0.6526716
time: 3.76 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.4761906, -9.1997719, -11.4838705, -9.1912203, -1.6449125, 1.6479900
1: -6.5077677, -4.7197447, -6.5255523, -4.7098818, -1.6360612, 1.6427562
2: -6.2201405, -4.1730690, -6.2346706, -4.1530495, -1.6440022, 1.6368639
3: -5.3434172, -3.7515755, -5.3562336, -3.7392907, -1.2368944, 1.2388295
4: -7.3794909, -5.1486406, -7.4012957, -5.1320043, -1.5959651, 1.5938227
5: -10.4566364, -8.5936108, -10.4910154, -8.5676641, -1.3438067, 1.3546557
6: -17.1231861, -14.7014198, -17.1370125, -14.6826668, -1.6652322, 1.6520991
7: 5.0322008, 6.2533360, 5.0215483, 6.2595429, -1.0949090, 1.0962446
8: -6.4444623, -4.6449203, -6.4545035, -4.6381946, -1.3515844, 1.3544888
9: -5.4740334, -3.8043919, -5.4961004, -3.7896469, -1.4929838, 1.5017676

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6423698, upper bound: 0.6517349
time: 7.17 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455077, upper bound: 0.6517349
time: 5.32 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.4845028, -9.1906242, -11.4845047, -9.1906223, -1.6583149, 1.6527426
1: -6.5281105, -4.7085719, -6.5281162, -4.7085686, -1.6592188, 1.6574290
2: -6.2392993, -4.1525431, -6.2393064, -4.1525431, -1.6564403, 1.6516892
3: -5.3599296, -3.7378807, -5.3599386, -3.7378786, -1.2537526, 1.2512122
4: -7.4078784, -5.1316471, -7.4078903, -5.1316462, -1.6123013, 1.6121126
5: -10.4926329, -8.5582266, -10.4926348, -8.5582123, -1.3681912, 1.3674654
6: -17.1419468, -14.6815910, -17.1419563, -14.6815891, -1.6752982, 1.6643901
7: 5.0200725, 6.2601008, 5.0200686, 6.2601018, -1.1024842, 1.1023668
8: -6.4556746, -4.6370997, -6.4556761, -4.6370997, -1.3569298, 1.3685844
9: -5.4970646, -3.7843776, -5.4970665, -3.7843659, -1.5138388, 1.5089753

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455054, upper bound: 0.6495325
time: 4.56 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455054, upper bound: 0.6526714
time: 4.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.73 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6422902, upper bound: 0.6445075
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6454272, upper bound: 0.6445078
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6454274, upper bound: 0.6422903
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6454274, upper bound: 0.6454275
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6495325, upper bound: 0.6445070
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6526704, upper bound: 0.6445072
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6526706, upper bound: 0.6422900
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6526706, upper bound: 0.6454269
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6445069, upper bound: 0.6495325
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6445069, upper bound: 0.6526703
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6422895, upper bound: 0.6526706
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6454266, upper bound: 0.6526716
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6423698, upper bound: 0.6517349
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6455077, upper bound: 0.6517349
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6455054, upper bound: 0.6495325
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.73
Output dim: 7, lower bound: -0.6455054, upper bound: 0.6526714

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.4630165, -9.2571163, -11.4671249, -9.2540417, -1.5947895, 1.5978870
1: -6.5030842, -4.7265863, -6.5131626, -4.7201996, -1.6148996, 1.6149724
2: -6.2067380, -4.2399020, -6.2192492, -4.2208843, -1.5982842, 1.5915861
3: -5.3389597, -3.7640829, -5.3346481, -3.7532554, -1.2179291, 1.2038528
4: -7.3735528, -5.1655774, -7.3923373, -5.1585908, -1.5671926, 1.5695528
5: -10.4457083, -8.6375017, -10.4757061, -8.6140156, -1.3156705, 1.3178647
6: -17.1152649, -14.7268171, -17.1268482, -14.7190886, -1.6267154, 1.6227875
7: 5.0619850, 6.2475471, 5.0540390, 6.2491217, -1.0645952, 1.0690078
8: -6.4353361, -4.6829929, -6.4314017, -4.6772189, -1.3114266, 1.3021982
9: -5.4283996, -3.8126690, -5.4454541, -3.8074172, -1.4384305, 1.4512815

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422909, upper bound: 0.6413907
time: 6.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422884, upper bound: 0.6445074
time: 4.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.4630136, -9.2571259, -11.5206385, -9.2455282, -1.6084843, 1.6349050
1: -6.5030727, -4.7265902, -6.5345483, -4.6684875, -1.6442757, 1.6554856
2: -6.2067356, -4.2399035, -6.2457490, -4.2165480, -1.6080556, 1.6173865
3: -5.3389506, -3.7640848, -5.3610945, -3.6857200, -1.2349309, 1.2375307
4: -7.3735495, -5.1655846, -7.4429197, -5.1467171, -1.5842967, 1.5955453
5: -10.4457064, -8.6375036, -10.4851837, -8.5871248, -1.3305295, 1.3271673
6: -17.1152611, -14.7268305, -17.1790009, -14.7014151, -1.6482215, 1.6526164
7: 5.0619884, 6.2475452, 5.0289869, 6.2561116, -1.0726388, 1.0876591
8: -6.4353237, -4.6829939, -6.4490542, -4.6258783, -1.3403938, 1.3224168
9: -5.4283962, -3.8126752, -5.4917984, -3.7915747, -1.4581194, 1.4892738

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454255, upper bound: 0.6428322
time: 4.14 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6454255, upper bound: 0.6445025
time: 4.52 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -11.4677677, -9.2534475, -11.4713373, -9.2479610, -1.6158009, 1.6057885
1: -6.5157337, -4.7188835, -6.5234861, -4.7154131, -1.6314135, 1.6362317
2: -6.2238746, -4.2203856, -6.2258945, -4.2193699, -1.6174688, 1.6147590
3: -5.3383589, -3.7518444, -5.3554745, -3.7503874, -1.2190056, 1.2366804
4: -7.3989215, -5.1582298, -7.4019461, -5.1485791, -1.5880046, 1.5904067
5: -10.4773207, -8.6045771, -10.4817085, -8.6021051, -1.3442004, 1.3367918
6: -17.1317749, -14.7180109, -17.1340332, -14.7069740, -1.6558514, 1.6379125
7: 5.0525742, 6.2496819, 5.0498319, 6.2543144, -1.0799198, 1.0779641
8: -6.4325743, -4.6761236, -6.4465556, -4.6751704, -1.3083355, 1.3423314
9: -5.4464297, -3.8021324, -5.4514380, -3.7926483, -1.4729457, 1.4544308

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422909, upper bound: 0.6422877
time: 7.03 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422884, upper bound: 0.6422883
time: 5.36 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -11.5212708, -9.2449646, -11.4713345, -9.2479706, -1.6450627, 1.6194050
1: -6.5370464, -4.6672096, -6.5234756, -4.7154164, -1.6721396, 1.6648040
2: -6.2503781, -4.2160397, -6.2258925, -4.2193713, -1.6369362, 1.6213751
3: -5.3647909, -3.6843188, -5.3554659, -3.7503877, -1.2526612, 1.2482897
4: -7.4494863, -5.1463513, -7.4019423, -5.1485844, -1.6087563, 1.6075033
5: -10.4868393, -8.5776787, -10.4817066, -8.6021080, -1.3534589, 1.3541898
6: -17.1838608, -14.7003584, -17.1340294, -14.7069912, -1.6756890, 1.6593221
7: 5.0275116, 6.2566938, 5.0498362, 6.2543116, -1.0938663, 1.0860296
8: -6.4502234, -4.6247826, -6.4465432, -4.6751714, -1.3285232, 1.3573750
9: -5.4927716, -3.7863069, -5.4514341, -3.7926555, -1.4963372, 1.4739008

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6153

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445100, upper bound: 0.6454266
time: 6.40 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445075, upper bound: 0.6454296
time: 4.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.4630165, -9.2571163, -11.4803028, -9.1966972, -1.6261377, 1.6113565
1: -6.5030842, -4.7265863, -6.5178065, -4.7133512, -1.6222594, 1.6191018
2: -6.2067380, -4.2399020, -6.2326550, -4.1540575, -1.6290567, 1.6045618
3: -5.3389597, -3.7640829, -5.3391247, -3.7407470, -1.2290256, 1.2086071
4: -7.3735528, -5.1655774, -7.3982773, -5.1416569, -1.5848126, 1.5746579
5: -10.4457083, -8.6375017, -10.4866333, -8.5701256, -1.3305178, 1.3288109
6: -17.1152649, -14.7268171, -17.1347694, -14.6937180, -1.6452982, 1.6309798
7: 5.0619850, 6.2475471, 5.0242705, 6.2549086, -1.0704045, 1.0875443
8: -6.4353361, -4.6829929, -6.4405227, -4.6391468, -1.3407884, 1.3112311
9: -5.4283996, -3.8126690, -5.4910936, -3.7991278, -1.4467659, 1.4930745

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6495307, upper bound: 0.6413913
time: 9.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6495307, upper bound: 0.6445072
time: 9.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.4630136, -9.2571259, -11.5338020, -9.1881857, -1.6387601, 1.6483483
1: -6.5030727, -4.7265902, -6.5391197, -4.6616502, -1.6469359, 1.6596575
2: -6.2067356, -4.2399035, -6.2591596, -4.1497216, -1.6349072, 1.6303554
3: -5.3389506, -3.7640848, -5.3655329, -3.6732104, -1.2386425, 1.2422397
4: -7.3735495, -5.1655846, -7.4488668, -5.1297770, -1.6019063, 1.6006413
5: -10.4457064, -8.6375036, -10.4961109, -8.5432453, -1.3387296, 1.3381147
6: -17.1152611, -14.7268305, -17.1869202, -14.6760159, -1.6669290, 1.6608158
7: 5.0619884, 6.2475452, 4.9992304, 6.2618980, -1.0784473, 1.0972295
8: -6.4353237, -4.6829939, -6.4581766, -4.5878105, -1.3501003, 1.3314474
9: -5.4283962, -3.8126752, -5.5374146, -3.7833002, -1.4664414, 1.5062348

Time for backsubstitution: 12.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526662, upper bound: 0.6428333
time: 3.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6526662, upper bound: 0.6445048
time: 4.53 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -11.4677677, -9.2534475, -11.4845047, -9.1906223, -1.6396716, 1.6192415
1: -6.5157337, -4.7188835, -6.5281162, -4.7085686, -1.6387751, 1.6404033
2: -6.2238746, -4.2203856, -6.2393064, -4.1525431, -1.6419940, 1.6277621
3: -5.3383589, -3.7518444, -5.3599386, -3.7378786, -1.2301021, 1.2414262
4: -7.3989215, -5.1582298, -7.4078903, -5.1316462, -1.6027992, 1.5954976
5: -10.4773207, -8.6045771, -10.4926348, -8.5582123, -1.3523970, 1.3477376
6: -17.1317749, -14.7180109, -17.1419563, -14.6815891, -1.6644802, 1.6461051
7: 5.0525742, 6.2496819, 5.0200686, 6.2601018, -1.0857296, 1.0906113
8: -6.4325743, -4.6761236, -6.4556761, -4.6370997, -1.3340073, 1.3513701
9: -5.4464297, -3.8021324, -5.4970665, -3.7843659, -1.4812727, 1.4934790

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6495307, upper bound: 0.6422879
time: 5.48 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6495332, upper bound: 0.6422881
time: 4.55 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -11.5212708, -9.2449646, -11.4845009, -9.1906300, -1.6616712, 1.6328578
1: -6.5370464, -4.6672096, -6.5281043, -4.7085724, -1.6770492, 1.6689944
2: -6.2503781, -4.2160397, -6.2393050, -4.1525450, -1.6581483, 1.6343945
3: -5.3647909, -3.6843188, -5.3599305, -3.7378795, -1.2633324, 1.2530667
4: -7.4494863, -5.1463513, -7.4078865, -5.1316533, -1.6154652, 1.6125941
5: -10.4868393, -8.5776787, -10.4926319, -8.5582161, -1.3616557, 1.3651460
6: -17.1838608, -14.7003584, -17.1419525, -14.6816053, -1.6804566, 1.6675146
7: 5.0275116, 6.2566938, 5.0200715, 6.2600994, -1.0996804, 1.0986580
8: -6.4502234, -4.6247826, -6.4556632, -4.6370993, -1.3543186, 1.3664317
9: -5.4927716, -3.7863069, -5.4970608, -3.7843709, -1.5046706, 1.5130203

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6153

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6517347, upper bound: 0.6454267
time: 6.80 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6517347, upper bound: 0.6454267
time: 5.40 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4803028, -9.1966972, -11.4630165, -9.2571163, -1.6113567, 1.6261377
1: -6.5178065, -4.7133512, -6.5030842, -4.7265863, -1.6191020, 1.6222596
2: -6.2326550, -4.1540575, -6.2067380, -4.2399020, -1.6045623, 1.6290562
3: -5.3391247, -3.7407470, -5.3389597, -3.7640829, -1.2086070, 1.2290256
4: -7.3982773, -5.1416569, -7.3735528, -5.1655774, -1.5746577, 1.5848125
5: -10.4866333, -8.5701256, -10.4457083, -8.6375017, -1.3288109, 1.3305178
6: -17.1347694, -14.6937180, -17.1152649, -14.7268171, -1.6309800, 1.6452978
7: 5.0242705, 6.2549086, 5.0619850, 6.2475471, -1.0875442, 1.0704045
8: -6.4405227, -4.6391468, -6.4353361, -4.6829929, -1.3112314, 1.3407885
9: -5.4910936, -3.7991278, -5.4283996, -3.8126690, -1.4930747, 1.4467659

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6413908, upper bound: 0.6495311
time: 3.99 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6413908, upper bound: 0.6495331
time: 4.22 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -11.5338020, -9.1881857, -11.4630136, -9.2571259, -1.6483483, 1.6387599
1: -6.5391197, -4.6616502, -6.5030727, -4.7265902, -1.6596572, 1.6469359
2: -6.2591596, -4.1497216, -6.2067356, -4.2399035, -1.6303554, 1.6349074
3: -5.3655329, -3.6732104, -5.3389506, -3.7640848, -1.2422397, 1.2386426
4: -7.4488668, -5.1297770, -7.3735495, -5.1655846, -1.6006413, 1.6019067
5: -10.4961109, -8.5432453, -10.4457064, -8.6375036, -1.3381147, 1.3387295
6: -17.1869202, -14.6760159, -17.1152611, -14.7268305, -1.6608157, 1.6669289
7: 4.9992304, 6.2618980, 5.0619884, 6.2475452, -1.0972295, 1.0784473
8: -6.4581766, -4.5878105, -6.4353237, -4.6829939, -1.3314471, 1.3501004
9: -5.5374146, -3.7833002, -5.4283962, -3.8126752, -1.5062346, 1.4664414

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6428309, upper bound: 0.6526686
time: 3.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445023, upper bound: 0.6526686
time: 3.83 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -11.4845047, -9.1906223, -11.4677677, -9.2534475, -1.6192412, 1.6396718
1: -6.5281162, -4.7085686, -6.5157337, -4.7188835, -1.6404033, 1.6387746
2: -6.2393064, -4.1525431, -6.2238746, -4.2203856, -1.6277618, 1.6419941
3: -5.3599386, -3.7378786, -5.3383589, -3.7518444, -1.2414262, 1.2301019
4: -7.4078903, -5.1316462, -7.3989215, -5.1582298, -1.5954976, 1.6027992
5: -10.4926348, -8.5582123, -10.4773207, -8.6045771, -1.3477378, 1.3523971
6: -17.1419563, -14.6815891, -17.1317749, -14.7180109, -1.6461051, 1.6644803
7: 5.0200686, 6.2601018, 5.0525742, 6.2496819, -1.0906112, 1.0857295
8: -6.4556761, -4.6370997, -6.4325743, -4.6761236, -1.3513699, 1.3340074
9: -5.4970665, -3.7843659, -5.4464297, -3.8021324, -1.4934793, 1.4812727

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422878, upper bound: 0.6495307
time: 5.27 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6422878, upper bound: 0.6526706
time: 6.32 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -11.4845009, -9.1906300, -11.5212708, -9.2449646, -1.6328578, 1.6616714
1: -6.5281043, -4.7085724, -6.5370464, -4.6672096, -1.6689947, 1.6770489
2: -6.2393050, -4.1525450, -6.2503781, -4.2160397, -1.6343944, 1.6581485
3: -5.3599305, -3.7378795, -5.3647909, -3.6843188, -1.2530665, 1.2633322
4: -7.4078865, -5.1316533, -7.4494863, -5.1463513, -1.6125941, 1.6154652
5: -10.4926319, -8.5582161, -10.4868393, -8.5776787, -1.3651462, 1.3616554
6: -17.1419525, -14.6816053, -17.1838608, -14.7003584, -1.6675148, 1.6804565
7: 5.0200715, 6.2600994, 5.0275116, 6.2566938, -1.0986581, 1.0996804
8: -6.4556632, -4.6370993, -6.4502234, -4.6247826, -1.3664317, 1.3543186
9: -5.4970608, -3.7843709, -5.4927716, -3.7863069, -1.5130205, 1.5046705

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445067, upper bound: 0.6517372
time: 3.67 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445067, upper bound: 0.6526707
time: 4.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.4761906, -9.1997719, -11.4803028, -9.1966972, -1.6395760, 1.6428401
1: -6.5077677, -4.7197447, -6.5178065, -4.7133512, -1.6329257, 1.6329226
2: -6.2201405, -4.1730690, -6.2326550, -4.1540575, -1.6420691, 1.6354288
3: -5.3434172, -3.7515755, -5.3391247, -3.7407470, -1.2363987, 1.2219311
4: -7.3794909, -5.1486406, -7.3982773, -5.1416569, -1.5864704, 1.5888395
5: -10.4566364, -8.5936108, -10.4866333, -8.5701256, -1.3414769, 1.3498389
6: -17.1231861, -14.7014198, -17.1347694, -14.6937180, -1.6534970, 1.6495233
7: 5.0322008, 6.2533360, 5.0242705, 6.2549086, -1.0889800, 1.0933585
8: -6.4444623, -4.6449203, -6.4405227, -4.6391468, -1.3498299, 1.3406166
9: -5.4740334, -3.8043919, -5.4910936, -3.7991278, -1.4836662, 1.4964933

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6423655, upper bound: 0.6486209
time: 5.85 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6423655, upper bound: 0.6517349
time: 5.46 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.4761868, -9.1997805, -11.5338020, -9.1881857, -1.6521978, 1.6649516
1: -6.5077581, -4.7197461, -6.5391197, -4.6616502, -1.6547499, 1.6710377
2: -6.2201390, -4.1730709, -6.2591596, -4.1497216, -1.6479201, 1.6515698
3: -5.3434067, -3.7515757, -5.3655329, -3.6732104, -1.2443755, 1.2557086
4: -7.3794885, -5.1486473, -7.4488668, -5.1297770, -1.6035643, 1.6078157
5: -10.4566326, -8.5936146, -10.4961109, -8.5432453, -1.3496890, 1.3591938
6: -17.1231842, -14.7014313, -17.1869202, -14.6760159, -1.6751282, 1.6655707
7: 5.0322046, 6.2533321, 4.9992304, 6.2618980, -1.0970535, 1.1030437
8: -6.4444494, -4.6449203, -6.4581766, -4.5878105, -1.3591418, 1.3609333
9: -5.4740291, -3.8043985, -5.5374146, -3.7833002, -1.5033422, 1.5145595

Time for backsubstitution: 12.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455011, upper bound: 0.6500576
time: 3.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6455010, upper bound: 0.6517331
time: 3.68 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -11.4809475, -9.1961040, -11.4845047, -9.1906223, -1.6531174, 1.6473942
1: -6.5203629, -4.7120352, -6.5281162, -4.7085686, -1.6494370, 1.6542544
2: -6.2372808, -4.1535563, -6.2393064, -4.1525431, -1.6549916, 1.6497509
3: -5.3428335, -3.7393367, -5.3599386, -3.7378786, -1.2368795, 1.2507232
4: -7.4048653, -5.1412950, -7.4078903, -5.1316462, -1.6072888, 1.6026702
5: -10.4882479, -8.5606852, -10.4926348, -8.5582123, -1.3633537, 1.3651515
6: -17.1396980, -14.6926422, -17.1419563, -14.6815891, -1.6726794, 1.6526586
7: 5.0228057, 6.2554684, 5.0200686, 6.2601018, -1.0995848, 1.0964252
8: -6.4416914, -4.6380496, -6.4556761, -4.6370997, -1.3430531, 1.3668306
9: -5.4920678, -3.7938507, -5.4970665, -3.7843659, -1.5085154, 1.4996014

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6423655, upper bound: 0.6495310
time: 4.91 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6423655, upper bound: 0.6495311
time: 6.48 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -11.5344324, -9.1876249, -11.4845009, -9.1906300, -1.6751161, 1.6599712
1: -6.5416040, -4.6603708, -6.5281043, -4.7085724, -1.6848559, 1.6752844
2: -6.2637897, -4.1492090, -6.2393050, -4.1525450, -1.6711221, 1.6555972
3: -5.3692269, -3.6718078, -5.3599305, -3.7378795, -1.2706535, 1.2577424
4: -7.4554334, -5.1294107, -7.4078865, -5.1316533, -1.6210253, 1.6197709
5: -10.4977665, -8.5337992, -10.4926319, -8.5582161, -1.3726125, 1.3733463
6: -17.1917839, -14.6749582, -17.1419525, -14.6816053, -1.6886559, 1.6742877
7: 4.9977570, 6.2624803, 5.0200715, 6.2600994, -1.1092420, 1.1044718
8: -6.4593439, -4.5867138, -6.4556632, -4.6370993, -1.3633623, 1.3761383
9: -5.5383854, -3.7780387, -5.4970608, -3.7843709, -1.5216289, 1.5190594

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6153

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445699, upper bound: 0.6526698
time: 5.60 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6445699, upper bound: 0.6526733
time: 6.31 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.70 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6422909, upper bound: 0.6413907
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6422884, upper bound: 0.6445074
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6454255, upper bound: 0.6428322
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6454255, upper bound: 0.6445025
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6422909, upper bound: 0.6422877
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6422884, upper bound: 0.6422883
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6445100, upper bound: 0.6454266
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6445075, upper bound: 0.6454296
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6495307, upper bound: 0.6413913
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6495307, upper bound: 0.6445072
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6526662, upper bound: 0.6428333
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6526662, upper bound: 0.6445048
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6495307, upper bound: 0.6422879
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6495332, upper bound: 0.6422881
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6517347, upper bound: 0.6454267
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6517347, upper bound: 0.6454267
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6413908, upper bound: 0.6495311
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6413908, upper bound: 0.6495331
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6428309, upper bound: 0.6526686
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6445023, upper bound: 0.6526686
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6422878, upper bound: 0.6495307
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6422878, upper bound: 0.6526706
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6445067, upper bound: 0.6517372
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6445067, upper bound: 0.6526707
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6423655, upper bound: 0.6486209
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6423655, upper bound: 0.6517349
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6455011, upper bound: 0.6500576
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6455010, upper bound: 0.6517331
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6423655, upper bound: 0.6495310
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6423655, upper bound: 0.6495311
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6445699, upper bound: 0.6526698
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.70
Output dim: 7, lower bound: -0.6445699, upper bound: 0.6526733

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4594135, -9.2625589, -11.4671249, -9.2540417, -1.5894632, 1.5925493
1: -6.4953718, -4.7301121, -6.5131626, -4.7201996, -1.6051159, 1.6117144
2: -6.2047482, -4.2409124, -6.2192492, -4.2208843, -1.5968351, 1.5896654
3: -5.3218083, -3.7655363, -5.3346481, -3.7532554, -1.2010026, 1.2033625
4: -7.3705215, -5.1752443, -7.3923373, -5.1585908, -1.5621696, 1.5600611
5: -10.4413052, -8.6400137, -10.4757061, -8.6140156, -1.3108349, 1.3154757
6: -17.1130810, -14.7378483, -17.1268482, -14.7190886, -1.6241291, 1.6111130
7: 5.0646706, 6.2428956, 5.0540390, 6.2491217, -1.0618534, 1.0630698
8: -6.4213705, -4.6839457, -6.4314017, -4.6772189, -1.2976584, 1.3004389
9: -5.4233603, -3.8221662, -5.4454541, -3.8074172, -1.4331436, 1.4419434

Time for backsubstitution: 12.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6153

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6413933, upper bound: 0.6413914
time: 4.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6413933, upper bound: 0.6413936
time: 3.98 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -11.5122852, -9.2540188, -11.4671249, -9.2540417, -1.6261880, 1.6028292
1: -6.5168595, -4.6784253, -6.5131626, -4.7201996, -1.6293054, 1.6410232
2: -6.2311125, -4.2366943, -6.2192492, -4.2208843, -1.6227338, 1.5947704
3: -5.3481035, -3.6989138, -5.3346481, -3.7532554, -1.2291987, 1.2183573
4: -7.4200454, -5.1633930, -7.3923373, -5.1585908, -1.5885196, 1.5734156
5: -10.4507265, -8.6135521, -10.4757061, -8.6140156, -1.3191905, 1.3361998
6: -17.1638985, -14.7201900, -17.1268482, -14.7190886, -1.6532922, 1.6303437
7: 5.0399957, 6.2498274, 5.0540390, 6.2491217, -1.0800610, 1.0708553
8: -6.4389791, -4.6356630, -6.4314017, -4.6772189, -1.3159213, 1.3278315
9: -5.4682379, -3.8064160, -5.4454541, -3.8074172, -1.4675233, 1.4584713

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6153

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6413933, upper bound: 0.6445079
time: 4.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6413933, upper bound: 0.6445079
time: 3.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -11.4610357, -9.2592316, -11.4897327, -9.2550144, -1.5977705, 1.5957212
1: -6.5008650, -4.7328944, -6.4991417, -4.6924105, -1.6157868, 1.6070778
2: -6.1999245, -4.2410207, -6.2191720, -4.2660580, -1.5553141, 1.5883496
3: -5.3326087, -3.7645297, -5.3358173, -3.7043965, -1.2058966, 1.2120715
4: -7.3722200, -5.1721773, -7.4092236, -5.1704254, -1.5590401, 1.5494664
5: -10.4444761, -8.6422253, -10.4550629, -8.6046295, -1.3096733, 1.2935503
6: -17.1122360, -14.7285480, -17.1642723, -14.7183523, -1.6234837, 1.6312065
7: 5.0643034, 6.2466006, 5.0397406, 6.2401638, -1.0522718, 1.0748389
8: -6.4312658, -4.6845779, -6.4257727, -4.6393189, -1.3183336, 1.2980504
9: -5.4267383, -3.8278575, -5.4481530, -3.8451827, -1.4014113, 1.4162331

Time for backsubstitution: 12.45 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.090501308441162
rel_dist={7: [-0.6526852532184364, 0.6526854163544584]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 577

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741247, upper bound: 0.4689571
time: 4.26 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741525, upper bound: 0.4741514
time: 4.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.89
Output dim: 7, lower bound: -0.4741247, upper bound: 0.4689571
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.89
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

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689544, upper bound: 0.4689541
time: 3.49 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689571, upper bound: 0.4689571
time: 6.06 seconds

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

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4735581, upper bound: 0.4741506
time: 6.21 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741515, upper bound: 0.4741503
time: 6.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.41 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 7, lower bound: -0.4689544, upper bound: 0.4689541
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 7, lower bound: -0.4689571, upper bound: 0.4689571
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 7, lower bound: -0.4735581, upper bound: 0.4741506
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 7, lower bound: -0.4741515, upper bound: 0.4741503

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

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689536, upper bound: 0.4683629
time: 3.63 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689537, upper bound: 0.4689531
time: 4.97 seconds

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

Time for backsubstitution: 13.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689564, upper bound: 0.4683597
time: 3.85 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689564, upper bound: 0.4689560
time: 4.15 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.4833937, -9.1916742, -11.4736671, -9.2555742, -1.3268390, 1.3480535
1: -6.5235806, -4.7108526, -6.5057321, -4.7264228, -1.4328163, 1.4239342
2: -6.2312627, -4.1534328, -6.2185121, -4.2385764, -1.3990345, 1.4359539
3: -5.3534641, -3.7403922, -5.3404799, -3.7606781, -1.0403199, 1.0481812
4: -7.3963852, -5.1322775, -7.3777184, -5.1652808, -1.3198457, 1.3394966
5: -10.4897451, -8.5747337, -10.4562502, -8.6355896, -1.1162376, 1.1161460
6: -17.1333427, -14.6834793, -17.1214256, -14.7258167, -1.3349180, 1.3605405
7: 5.0226717, 6.2591119, 5.0607843, 6.2531571, -0.9821775, 0.9641345
8: -6.4536066, -4.6390581, -6.4434628, -4.6814065, -1.1040933, 1.1266801
9: -5.4953566, -3.7935975, -5.4289122, -3.8052411, -1.3553412, 1.3086715

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4683602, upper bound: 0.4741234
time: 7.81 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4683602, upper bound: 0.4741234
time: 5.58 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.4845047, -9.1906233, -11.4819860, -9.2464170, -1.3335340, 1.3601730
1: -6.5281134, -4.7085695, -6.5261369, -4.7152519, -1.4491601, 1.4482393
2: -6.2393050, -4.1525421, -6.2376595, -4.2180419, -1.4236367, 1.4455900
3: -5.3599358, -3.7378788, -5.3569803, -3.7469819, -1.0595939, 1.0627782
4: -7.4078879, -5.1316457, -7.4060950, -5.1482811, -1.3443351, 1.3449939
5: -10.4926348, -8.5582190, -10.4922514, -8.6002083, -1.1296666, 1.1476545
6: -17.1419525, -14.6815910, -17.1401863, -14.7059755, -1.3574488, 1.3657446
7: 5.0200696, 6.2601018, 5.0486231, 6.2599211, -0.9882147, 0.9757005
8: -6.4556746, -4.6370983, -6.4546833, -4.6735849, -1.1299641, 1.1280313
9: -5.4970675, -3.7843676, -5.4519496, -3.7852366, -1.3564715, 1.3436923

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689534, upper bound: 0.4741229
time: 3.41 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689534, upper bound: 0.4741233
time: 4.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.55 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.55
Output dim: 7, lower bound: -0.4689536, upper bound: 0.4683629
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.55
Output dim: 7, lower bound: -0.4689537, upper bound: 0.4689531
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.55
Output dim: 7, lower bound: -0.4689564, upper bound: 0.4683597
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.55
Output dim: 7, lower bound: -0.4689564, upper bound: 0.4689560
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 20.55
Output dim: 7, lower bound: -0.4683602, upper bound: 0.4741234
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 20.55
Output dim: 7, lower bound: -0.4683602, upper bound: 0.4741234
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 20.55
Output dim: 7, lower bound: -0.4689534, upper bound: 0.4741229
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 20.55
Output dim: 7, lower bound: -0.4689534, upper bound: 0.4741233

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4630165, -9.2571163, -11.4702225, -9.2490139, -1.3106906, 1.3139729
1: -6.5030842, -4.7265863, -6.5189228, -4.7176986, -1.4132252, 1.4188595
2: -6.2067380, -4.2399020, -6.2178488, -4.2202568, -1.3944416, 1.3865557
3: -5.3389597, -3.7640829, -5.3489971, -3.7529013, -1.0341778, 1.0337816
4: -7.3735528, -5.1655774, -7.3904400, -5.1492081, -1.3184514, 1.3131166
5: -10.4457083, -8.6375017, -10.4788160, -8.6186218, -1.0946937, 1.1056228
6: -17.1152649, -14.7268171, -17.1254196, -14.7088661, -1.3393104, 1.3270791
7: 5.0619850, 6.2475471, 5.0524364, 6.2533236, -0.9588315, 0.9593635
8: -6.4353361, -4.6829929, -6.4444828, -4.6771288, -1.0941212, 1.0959506
9: -5.4283996, -3.8126690, -5.4497261, -3.8018909, -1.3011928, 1.3135955

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667869, upper bound: 0.4683309
time: 4.42 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689543, upper bound: 0.4683613
time: 4.53 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -11.4713373, -9.2479630, -11.4713364, -9.2479630, -1.3303728, 1.3206713
1: -6.5234799, -4.7154164, -6.5234838, -4.7154150, -1.4375238, 1.4351637
2: -6.2258859, -4.2193723, -6.2258925, -4.2193708, -1.4103751, 1.4111266
3: -5.3554649, -3.7503891, -5.3554716, -3.7503872, -1.0517857, 1.0530579
4: -7.4019337, -5.1485777, -7.4019408, -5.1485786, -1.3297021, 1.3391279
5: -10.4817057, -8.6021194, -10.4817076, -8.6021109, -1.1288743, 1.1190579
6: -17.1340256, -14.7069778, -17.1340313, -14.7069769, -1.3545110, 1.3495985
7: 5.0498333, 6.2543139, 5.0498323, 6.2543144, -0.9703844, 0.9712567
8: -6.4465537, -4.6751709, -6.4465570, -4.6751695, -1.1002109, 1.1218202
9: -5.4514370, -3.7926612, -5.4514380, -3.7926533, -1.3362308, 1.3185709

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689259, upper bound: 0.4667841
time: 6.86 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689518, upper bound: 0.4689512
time: 6.52 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.4630165, -9.2571163, -11.4833937, -9.1916742, -1.3371897, 1.3274298
1: -6.5030842, -4.7265863, -6.5235806, -4.7108526, -1.4205832, 1.4230132
2: -6.2067380, -4.2399020, -6.2312627, -4.1534328, -1.4239609, 1.3995428
3: -5.3389597, -3.7640829, -5.3534641, -3.7403922, -1.0452750, 1.0385321
4: -7.3735528, -5.1655774, -7.3963852, -5.1322775, -1.3360753, 1.3182172
5: -10.4457083, -8.6375017, -10.4897451, -8.5747337, -1.1055624, 1.1165719
6: -17.1152649, -14.7268171, -17.1333427, -14.6834793, -1.3542013, 1.3352715
7: 5.0619850, 6.2475471, 5.0226717, 6.2591119, -0.9646406, 0.9764180
8: -6.4353361, -4.6829929, -6.4536066, -4.6390581, -1.1184454, 1.1049886
9: -5.4283996, -3.8126690, -5.4953566, -3.7935975, -1.3095386, 1.3478800

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4719534, upper bound: 0.4683301
time: 4.54 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741209, upper bound: 0.4683608
time: 4.15 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.4713373, -9.2479630, -11.4845047, -9.1906233, -1.3493111, 1.3341246
1: -6.5234799, -4.7154164, -6.5281134, -4.7085695, -1.4448848, 1.4393346
2: -6.2258859, -4.2193723, -6.2393050, -4.1525421, -1.4335968, 1.4208856
3: -5.3554649, -3.7503891, -5.3599358, -3.7378788, -1.0604495, 1.0578041
4: -7.4019337, -5.1485777, -7.4078879, -5.1316457, -1.3416417, 1.3438257
5: -10.4817057, -8.6021194, -10.4926348, -8.5582190, -1.1370711, 1.1287391
6: -17.1340256, -14.7069778, -17.1419525, -14.6815910, -1.3594048, 1.3562098
7: 5.0498333, 6.2543139, 5.0200696, 6.2601018, -0.9761939, 0.9824553
8: -6.4465537, -4.6751709, -6.4556746, -4.6370983, -1.1197977, 1.1292428
9: -5.4514370, -3.7926612, -5.4970675, -3.7843676, -1.3445578, 1.3490050

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4740924, upper bound: 0.4667837
time: 4.25 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741211, upper bound: 0.4689507
time: 4.96 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -11.4833937, -9.1916742, -11.4630165, -9.2571163, -1.3274298, 1.3371899
1: -6.5235806, -4.7108526, -6.5030842, -4.7265863, -1.4230132, 1.4205835
2: -6.2312627, -4.1534328, -6.2067380, -4.2399020, -1.3995423, 1.4239609
3: -5.3534641, -3.7403922, -5.3389597, -3.7640829, -1.0385320, 1.0452751
4: -7.3963852, -5.1322775, -7.3735528, -5.1655774, -1.3182175, 1.3360753
5: -10.4897451, -8.5747337, -10.4457083, -8.6375017, -1.1165721, 1.1055624
6: -17.1333427, -14.6834793, -17.1152649, -14.7268171, -1.3352714, 1.3542011
7: 5.0226717, 6.2591119, 5.0619850, 6.2475471, -0.9764180, 0.9646406
8: -6.4536066, -4.6390581, -6.4353361, -4.6829929, -1.1049886, 1.1184454
9: -5.4953566, -3.7935975, -5.4283996, -3.8126690, -1.3478801, 1.3095384

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4683299, upper bound: 0.4719561
time: 3.66 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4683582, upper bound: 0.4741235
time: 3.63 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -11.4833937, -9.1916742, -11.4761906, -9.1997719, -1.3532505, 1.3499360
1: -6.5235806, -4.7108526, -6.5077677, -4.7197447, -1.4352412, 1.4296727
2: -6.2312627, -4.1534328, -6.2201405, -4.1730690, -1.4286015, 1.4364219
3: -5.3534641, -3.7403922, -5.3434172, -3.7515755, -1.0498661, 1.0514407
4: -7.3963852, -5.1322775, -7.3794909, -5.1486406, -1.3314385, 1.3367760
5: -10.4897451, -8.5747337, -10.4566364, -8.5936108, -1.1330094, 1.1159164
6: -17.1333427, -14.6834793, -17.1231861, -14.7014198, -1.3494344, 1.3617835
7: 5.0226717, 6.2591119, 5.0322008, 6.2533360, -0.9814312, 0.9809575
8: -6.4536066, -4.6390581, -6.4444623, -4.6449203, -1.1284401, 1.1265165
9: -5.4953566, -3.7935975, -5.4740334, -3.8043919, -1.3550429, 1.3407764

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4683299, upper bound: 0.4719807
time: 4.48 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4683582, upper bound: 0.4741482
time: 6.50 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -11.4845047, -9.1906233, -11.4713373, -9.2479630, -1.3341248, 1.3493112
1: -6.5281134, -4.7085695, -6.5234799, -4.7154164, -1.4393344, 1.4448845
2: -6.2393050, -4.1525421, -6.2258859, -4.2193723, -1.4208853, 1.4335968
3: -5.3599358, -3.7378788, -5.3554649, -3.7503891, -1.0578041, 1.0604495
4: -7.4078879, -5.1316457, -7.4019337, -5.1485777, -1.3438258, 1.3416417
5: -10.4926348, -8.5582190, -10.4817057, -8.6021194, -1.1287391, 1.1370710
6: -17.1419525, -14.6815910, -17.1340256, -14.7069778, -1.3562098, 1.3594048
7: 5.0200696, 6.2601018, 5.0498333, 6.2543139, -0.9824553, 0.9761939
8: -6.4556746, -4.6370983, -6.4465537, -4.6751709, -1.1292429, 1.1197977
9: -5.4970675, -3.7843676, -5.4514370, -3.7926612, -1.3490047, 1.3445578

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667837, upper bound: 0.4740922
time: 4.60 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689512, upper bound: 0.4741208
time: 3.51 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -11.4845047, -9.1906233, -11.4845028, -9.1906242, -1.3562393, 1.3620509
1: -6.5281134, -4.7085695, -6.5281105, -4.7085719, -1.4516091, 1.4539723
2: -6.2393050, -4.1525421, -6.2392993, -4.1525431, -1.4447987, 1.4460500
3: -5.3599358, -3.7378788, -5.3599296, -3.7378807, -1.0649301, 1.0660489
4: -7.4078879, -5.1316457, -7.4078784, -5.1316471, -1.3508601, 1.3466648
5: -10.4926348, -8.5582190, -10.4926329, -8.5582266, -1.1393681, 1.1474226
6: -17.1419525, -14.6815910, -17.1419468, -14.6815910, -1.3619242, 1.3669871
7: 5.0200696, 6.2601018, 5.0200725, 6.2601008, -0.9874682, 0.9871622
8: -6.4556746, -4.6370983, -6.4556746, -4.6370997, -1.1402564, 1.1278738
9: -5.4970675, -3.7843676, -5.4970646, -3.7843776, -1.3561597, 1.3631212

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667837, upper bound: 0.4741200
time: 5.52 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689512, upper bound: 0.4741209
time: 6.13 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.60 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4667869, upper bound: 0.4683309
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4689543, upper bound: 0.4683613
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4689259, upper bound: 0.4667841
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4689518, upper bound: 0.4689512
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4719534, upper bound: 0.4683301
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4741209, upper bound: 0.4683608
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4740924, upper bound: 0.4667837
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4741211, upper bound: 0.4689507
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4683299, upper bound: 0.4719561
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4683582, upper bound: 0.4741235
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4683299, upper bound: 0.4719807
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4683582, upper bound: 0.4741482
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4667837, upper bound: 0.4740922
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4689512, upper bound: 0.4741208
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4667837, upper bound: 0.4741200
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.60
Output dim: 7, lower bound: -0.4689512, upper bound: 0.4741209

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.4625435, -9.2578392, -11.4666367, -9.2544918, -1.3046889, 1.3079646
1: -6.5020485, -4.7270412, -6.5111809, -4.7211776, -1.4087462, 1.4086130
2: -6.2064743, -4.2400417, -6.2158422, -4.2212644, -1.3923335, 1.3848279
3: -5.3366814, -3.7642593, -5.3318672, -3.7543571, -1.0314362, 1.0168072
4: -7.3731470, -5.1668620, -7.3874226, -5.1588631, -1.3083112, 1.3068342
5: -10.4451199, -8.6378174, -10.4744349, -8.6210918, -1.0917249, 1.1005065
6: -17.1149788, -14.7282848, -17.1231804, -14.7199039, -1.3273213, 1.3229376
7: 5.0623393, 6.2469292, 5.0551529, 6.2486897, -0.9525495, 0.9556887
8: -6.4334760, -4.6831083, -6.4305038, -4.6780844, -1.0905280, 1.0819681
9: -5.4277711, -3.8139336, -5.4447055, -3.8113766, -1.2910538, 1.3070660

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667833, upper bound: 0.4661924
time: 4.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667833, upper bound: 0.4683308
time: 4.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.4630108, -9.2571316, -11.5197430, -9.2459555, -1.3163309, 1.3406522
1: -6.5030670, -4.7265911, -6.5325537, -4.6696644, -1.4374385, 1.4448867
2: -6.2067347, -4.2399044, -6.2422910, -4.2170377, -1.4016314, 1.4096185
3: -5.3389444, -3.7640853, -5.3583145, -3.6875539, -1.0481167, 1.0433903
4: -7.3735456, -5.1655893, -7.4371300, -5.1469965, -1.3210425, 1.3318583
5: -10.4457016, -8.6375055, -10.4838810, -8.5945940, -1.1030138, 1.1089873
6: -17.1152611, -14.7268410, -17.1742058, -14.7022629, -1.3462186, 1.3499752
7: 5.0619898, 6.2475424, 5.0304947, 6.2556643, -0.9606736, 0.9733630
8: -6.4353147, -4.6829939, -6.4481316, -4.6291285, -1.1150579, 1.0998604
9: -5.4283924, -3.8126783, -5.4899335, -3.7955463, -1.3074896, 1.3379405

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689492, upper bound: 0.4674727
time: 3.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689492, upper bound: 0.4683557
time: 4.44 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -11.4677677, -9.2534475, -11.4708691, -9.2486877, -1.3243957, 1.3146107
1: -6.5157337, -4.7188835, -6.5224442, -4.7158628, -1.4273257, 1.4306767
2: -6.2238746, -4.2203856, -6.2256265, -4.2195101, -1.4086390, 1.4089996
3: -5.3383589, -3.7518444, -5.3531990, -3.7505660, -1.0348365, 1.0503206
4: -7.3989215, -5.1582298, -7.4015388, -5.1498604, -1.3234265, 1.3290389
5: -10.4773207, -8.6045771, -10.4811201, -8.6024218, -1.1237395, 1.1161048
6: -17.1317749, -14.7180109, -17.1337376, -14.7084465, -1.3502455, 1.3375256
7: 5.0525742, 6.2496819, 5.0501938, 6.2536993, -0.9666462, 0.9649632
8: -6.4325743, -4.6761236, -6.4446907, -4.6752830, -1.0862029, 1.1182299
9: -5.4464297, -3.8021324, -5.4508114, -3.7939134, -1.3297009, 1.3083804

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667833, upper bound: 0.4667833
time: 3.66 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667861, upper bound: 0.4667860
time: 4.13 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -11.5208588, -9.2449665, -11.4713306, -9.2479763, -1.3493609, 1.3261669
1: -6.5369768, -4.6674361, -6.5234671, -4.7154193, -1.4639692, 1.4580026
2: -6.2503185, -4.2161407, -6.2258883, -4.2193727, -1.4269867, 1.4144447
3: -5.3647790, -3.6850567, -5.3554559, -3.7503905, -1.0613828, 1.0615894
4: -7.4485989, -5.1463528, -7.4019356, -5.1485920, -1.3422585, 1.3417406
5: -10.4868355, -8.5780697, -10.4817047, -8.6021147, -1.1321611, 1.1264570
6: -17.1826820, -14.7004042, -17.1340237, -14.7070007, -1.3674555, 1.3564893
7: 5.0278950, 6.2566938, 5.0498366, 6.2543097, -0.9790351, 0.9731407
8: -6.4501953, -4.6271696, -6.4465365, -4.6751714, -1.1040549, 1.1287796
9: -5.4916525, -3.7863286, -5.4514294, -3.7926645, -1.3459854, 1.3245442

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4680659, upper bound: 0.4667811
time: 6.86 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689494, upper bound: 0.4667807
time: 4.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.4625435, -9.2578392, -11.4798145, -9.1971493, -1.3311939, 1.3214355
1: -6.5020485, -4.7270412, -6.5158367, -4.7143297, -1.4161034, 1.4127536
2: -6.2064743, -4.2400417, -6.2292471, -4.1544404, -1.4218488, 1.3978081
3: -5.3366814, -3.7642593, -5.3363447, -3.7418489, -1.0425327, 1.0215635
4: -7.3731470, -5.1668620, -7.3933649, -5.1419277, -1.3259315, 1.3119321
5: -10.4451199, -8.6378174, -10.4853640, -8.5772018, -1.1024826, 1.1114559
6: -17.1149788, -14.7282848, -17.1311035, -14.6945314, -1.3421421, 1.3311301
7: 5.0623393, 6.2469292, 5.0253835, 6.2544765, -0.9583588, 0.9726303
8: -6.4334760, -4.6831083, -6.4396286, -4.6400113, -1.1145051, 1.0910053
9: -5.4277711, -3.8139336, -5.4903455, -3.8030837, -1.2993968, 1.3410881

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4719526, upper bound: 0.4661918
time: 4.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4719526, upper bound: 0.4683300
time: 4.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.4630108, -9.2571316, -11.5329094, -9.1886120, -1.3417666, 1.3511354
1: -6.5030670, -4.7265911, -6.5371346, -4.6628284, -1.4400973, 1.4490192
2: -6.2067347, -4.2399044, -6.2556944, -4.1502113, -1.4272962, 1.4193318
3: -5.3389444, -3.7640853, -5.3627529, -3.6750445, -1.0518289, 1.0481017
4: -7.3735456, -5.1655893, -7.4430766, -5.1300545, -1.3386514, 1.3365562
5: -10.4457016, -8.6375055, -10.4948101, -8.5507154, -1.1112132, 1.1199372
6: -17.1152611, -14.7268410, -17.1821270, -14.6768570, -1.3611419, 1.3565867
7: 5.0619898, 6.2475424, 5.0007358, 6.2614498, -0.9664829, 0.9829396
8: -6.4353147, -4.6829939, -6.4572568, -4.5910597, -1.1247640, 1.1088951
9: -5.4283924, -3.8126783, -5.5355511, -3.7872679, -1.3158197, 1.3549080

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741185, upper bound: 0.4674694
time: 4.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741185, upper bound: 0.4683553
time: 5.66 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -11.4677677, -9.2534475, -11.4840403, -9.1913500, -1.3432786, 1.3280661
1: -6.5157337, -4.7188835, -6.5270748, -4.7090158, -1.4346864, 1.4348528
2: -6.2238746, -4.2203856, -6.2390370, -4.1526814, -1.4318781, 1.4187570
3: -5.3383589, -3.7518444, -5.3576641, -3.7380579, -1.0435033, 1.0550673
4: -7.3989215, -5.1582298, -7.4074826, -5.1329279, -1.3351648, 1.3337364
5: -10.4773207, -8.6045771, -10.4920464, -8.5585318, -1.1319358, 1.1256835
6: -17.1317749, -14.7180109, -17.1416626, -14.6830616, -1.3550138, 1.3441375
7: 5.0525742, 6.2496819, 5.0204296, 6.2594852, -0.9724553, 0.9761382
8: -6.4325743, -4.6761236, -6.4538107, -4.6372137, -1.1057155, 1.1253027
9: -5.4464297, -3.8021324, -5.4964428, -3.7856288, -1.3380277, 1.3388205

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4719526, upper bound: 0.4667828
time: 4.66 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4719526, upper bound: 0.4667856
time: 3.59 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -11.5208588, -9.2449665, -11.4844990, -9.1906385, -1.3659692, 1.3396198
1: -6.5369768, -4.6674361, -6.5280976, -4.7085748, -1.4668562, 1.4619545
2: -6.2503185, -4.2161407, -6.2393007, -4.1525445, -1.4481997, 1.4242034
3: -5.3647790, -3.6850567, -5.3599200, -3.7378812, -1.0701973, 1.0660044
4: -7.4485989, -5.1463528, -7.4078808, -5.1316595, -1.3489678, 1.3464390
5: -10.4868355, -8.5780697, -10.4926291, -8.5582237, -1.1403575, 1.1343760
6: -17.1826820, -14.7004042, -17.1419487, -14.6816149, -1.3722234, 1.3631010
7: 5.0278950, 6.2566938, 5.0200739, 6.2600965, -0.9832282, 0.9842956
8: -6.4501953, -4.6271696, -6.4556561, -4.6371012, -1.1236875, 1.1355578
9: -5.4916525, -3.7863286, -5.4970593, -3.7843781, -1.3521910, 1.3550534

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732326, upper bound: 0.4689487
time: 4.74 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4741187, upper bound: 0.4689484
time: 4.38 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4798145, -9.1971493, -11.4625435, -9.2578392, -1.3214352, 1.3311939
1: -6.5158367, -4.7143297, -6.5020485, -4.7270412, -1.4127538, 1.4161034
2: -6.2292471, -4.1544404, -6.2064743, -4.2400417, -1.3978081, 1.4218490
3: -5.3363447, -3.7418489, -5.3366814, -3.7642593, -1.0215635, 1.0425328
4: -7.3933649, -5.1419277, -7.3731470, -5.1668620, -1.3119321, 1.3259314
5: -10.4853640, -8.5772018, -10.4451199, -8.6378174, -1.1114559, 1.1024827
6: -17.1311035, -14.6945314, -17.1149788, -14.7282848, -1.3311298, 1.3421420
7: 5.0253835, 6.2544765, 5.0623393, 6.2469292, -0.9726303, 0.9583588
8: -6.4396286, -4.6400113, -6.4334760, -4.6831083, -1.0910056, 1.1145051
9: -5.4903455, -3.8030837, -5.4277711, -3.8139336, -1.3410881, 1.2993968

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661916, upper bound: 0.4719552
time: 3.36 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4661916, upper bound: 0.4719525
time: 5.47 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -11.5329094, -9.1886120, -11.4630108, -9.2571316, -1.3511353, 1.3417665
1: -6.5371346, -4.6628284, -6.5030670, -4.7265911, -1.4490194, 1.4400973
2: -6.2556944, -4.1502113, -6.2067347, -4.2399044, -1.4193316, 1.4272962
3: -5.3627529, -3.6750445, -5.3389444, -3.7640853, -1.0481017, 1.0518287
4: -7.4430766, -5.1300545, -7.3735456, -5.1655893, -1.3365562, 1.3386514
5: -10.4948101, -8.5507154, -10.4457016, -8.6375055, -1.1199367, 1.1112132
6: -17.1821270, -14.6768570, -17.1152611, -14.7268410, -1.3565867, 1.3611419
7: 5.0007358, 6.2614498, 5.0619898, 6.2475424, -0.9829395, 0.9664829
8: -6.4572568, -4.5910597, -6.4353147, -4.6829939, -1.1088951, 1.1247640
9: -5.5355511, -3.7872679, -5.4283924, -3.8126783, -1.3549080, 1.3158197

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4674696, upper bound: 0.4741211
time: 4.08 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4683555, upper bound: 0.4741211
time: 3.75 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -11.4798145, -9.1971493, -11.4757223, -9.2004929, -1.3472614, 1.3439420
1: -6.5158367, -4.7143297, -6.5067320, -4.7201982, -1.4249957, 1.4252009
2: -6.2292471, -4.1544404, -6.2198758, -4.1732073, -1.4268939, 1.4343086
3: -5.3363447, -3.7418489, -5.3411393, -3.7517519, -1.0328888, 1.0483406
4: -7.3933649, -5.1419277, -7.3790851, -5.1499233, -1.3251562, 1.3266315
5: -10.4853640, -8.5772018, -10.4560461, -8.5939302, -1.1278801, 1.1128372
6: -17.1311035, -14.6945314, -17.1229019, -14.7028904, -1.3450799, 1.3497247
7: 5.0253835, 6.2544765, 5.0325527, 6.2527175, -0.9776435, 0.9746665
8: -6.4396286, -4.6400113, -6.4426012, -4.6450343, -1.1143641, 1.1225760
9: -5.4903455, -3.8030837, -5.4734054, -3.8056560, -1.3482502, 1.3305917

Time for backsubstitution: 13.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4663201, upper bound: 0.4719836
time: 4.10 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4663201, upper bound: 0.4719807
time: 5.76 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -11.5329094, -9.1886120, -11.4761839, -9.1997862, -1.3700080, 1.3545123
1: -6.5371346, -4.6628284, -6.5077524, -4.7197485, -1.4575582, 1.4470429
2: -6.2556944, -4.1502113, -6.2201376, -4.1730719, -1.4432533, 1.4397568
3: -5.3627529, -3.6750445, -5.3434014, -3.7515762, -1.0595566, 1.0574203
4: -7.4430766, -5.1300545, -7.3794861, -5.1486521, -1.3435941, 1.3393512
5: -10.4948101, -8.5507154, -10.4566317, -8.5936165, -1.1364264, 1.1215675
6: -17.1821270, -14.6768570, -17.1231842, -14.7014427, -1.3623121, 1.3687246
7: 5.0007358, 6.2614498, 5.0322056, 6.2533298, -0.9879525, 0.9828191
8: -6.4572568, -4.5910597, -6.4444432, -4.6449213, -1.1323490, 1.1328349
9: -5.5355511, -3.7872679, -5.4740267, -3.8044016, -1.3620701, 1.3469410

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4675980, upper bound: 0.4741462
time: 4.13 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4684839, upper bound: 0.4741482
time: 3.90 seconds

## BFS IS instance: IS_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -11.4840403, -9.1913500, -11.4677677, -9.2534475, -1.3280659, 1.3432786
1: -6.5270748, -4.7090158, -6.5157337, -4.7188835, -1.4348521, 1.4346864
2: -6.2390370, -4.1526814, -6.2238746, -4.2203856, -1.4187567, 1.4318783
3: -5.3576641, -3.7380579, -5.3383589, -3.7518444, -1.0550673, 1.0435032
4: -7.4074826, -5.1329279, -7.3989215, -5.1582298, -1.3337364, 1.3351648
5: -10.4920464, -8.5585318, -10.4773207, -8.6045771, -1.1256833, 1.1319358
6: -17.1416626, -14.6830616, -17.1317749, -14.7180109, -1.3441374, 1.3550141
7: 5.0204296, 6.2594852, 5.0525742, 6.2496819, -0.9761384, 0.9724553
8: -6.4538107, -4.6372137, -6.4325743, -4.6761236, -1.1253026, 1.1057155
9: -5.4964428, -3.7856288, -5.4464297, -3.8021324, -1.3388200, 1.3380275

Time for backsubstitution: 13.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667829, upper bound: 0.4719524
time: 10.46 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667829, upper bound: 0.4740922
time: 8.69 seconds

## BFS IS instance: IS_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -11.4844990, -9.1906385, -11.5208588, -9.2449665, -1.3396196, 1.3659691
1: -6.5280976, -4.7085748, -6.5369768, -4.6674361, -1.4619544, 1.4668560
2: -6.2393007, -4.1525445, -6.2503185, -4.2161407, -1.4242032, 1.4481996
3: -5.3599200, -3.7378812, -5.3647790, -3.6850567, -1.0660043, 1.0701973
4: -7.4078808, -5.1316595, -7.4485989, -5.1463528, -1.3464389, 1.3489678
5: -10.4926291, -8.5582237, -10.4868355, -8.5780697, -1.1343760, 1.1403576
6: -17.1419487, -14.6816149, -17.1826820, -14.7004042, -1.3631010, 1.3722233
7: 5.0200739, 6.2600965, 5.0278950, 6.2566938, -0.9842956, 0.9832284
8: -6.4556561, -4.6371012, -6.4501953, -4.6271696, -1.1355577, 1.1236877
9: -5.4970593, -3.7843781, -5.4916525, -3.7863286, -1.3550534, 1.3521910

Time for backsubstitution: 12.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689487, upper bound: 0.4732324
time: 4.50 seconds

## Relational analysis of IS_A2_B2_B1_B2_B2

### Relational analysis result of IS_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4689488, upper bound: 0.4741186
time: 6.54 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -11.4840403, -9.1913500, -11.4809475, -9.1961040, -1.3502250, 1.3560320
1: -6.5270748, -4.7090158, -6.5203629, -4.7120352, -1.4471352, 1.4437807
2: -6.2390370, -4.1526814, -6.2372808, -4.1535563, -1.4426696, 1.4443238
3: -5.3576641, -3.7380579, -5.3428335, -3.7393367, -1.0618306, 1.0491078
4: -7.4074826, -5.1329279, -7.4048653, -5.1412950, -1.3407729, 1.3401868
5: -10.4920464, -8.5585318, -10.4882479, -8.5606852, -1.1363115, 1.1422873
6: -17.1416626, -14.6830616, -17.1396980, -14.6926422, -1.3498600, 1.3625968
7: 5.0204296, 6.2594852, 5.0228057, 6.2554684, -0.9811511, 0.9833528
8: -6.4538107, -4.6372137, -6.4416914, -4.6380496, -1.1363156, 1.1137909
9: -5.4964428, -3.7856288, -5.4920678, -3.7938507, -1.3459725, 1.3563273

Time for backsubstitution: 12.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4669114, upper bound: 0.4719807
time: 6.66 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4669114, upper bound: 0.4741205
time: 7.25 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -11.4844990, -9.1906385, -11.5340195, -9.1876259, -1.3607147, 1.3787212
1: -6.5280976, -4.7085748, -6.5415335, -4.6605968, -1.4676251, 1.4737949
2: -6.2393007, -4.1525445, -6.2637300, -4.1493101, -1.4481153, 1.4606211
3: -5.3599200, -3.7378812, -5.3692150, -3.6725469, -1.0709013, 1.0757565
4: -7.4078808, -5.1316595, -7.4545441, -5.1294117, -1.3534794, 1.3539982
5: -10.4926291, -8.5582237, -10.4977617, -8.5341873, -1.1450083, 1.1507097
6: -17.1419487, -14.6816149, -17.1906033, -14.6750050, -1.3688610, 1.3798063
7: 5.0200739, 6.2600965, 4.9981394, 6.2624803, -0.9893081, 0.9936094
8: -6.4556561, -4.6371012, -6.4593153, -4.5890989, -1.1465726, 1.1317604
9: -5.4970593, -3.7843781, -5.5372682, -3.7780614, -1.3621936, 1.3701166

Time for backsubstitution: 13.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690772, upper bound: 0.4732606
time: 4.55 seconds

## Relational analysis of IS_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4690773, upper bound: 0.4741468
time: 5.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.54 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4667833, upper bound: 0.4661924
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4667833, upper bound: 0.4683308
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4689492, upper bound: 0.4674727
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4689492, upper bound: 0.4683557
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4667833, upper bound: 0.4667833
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4667861, upper bound: 0.4667860
IS_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4680659, upper bound: 0.4667811
IS_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4689494, upper bound: 0.4667807
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4719526, upper bound: 0.4661918
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4719526, upper bound: 0.4683300
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4741185, upper bound: 0.4674694
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4741185, upper bound: 0.4683553
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4719526, upper bound: 0.4667828
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4719526, upper bound: 0.4667856
IS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4732326, upper bound: 0.4689487
IS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4741187, upper bound: 0.4689484
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4661916, upper bound: 0.4719552
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4661916, upper bound: 0.4719525
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4674696, upper bound: 0.4741211
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4683555, upper bound: 0.4741211
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4663201, upper bound: 0.4719836
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4663201, upper bound: 0.4719807
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4675980, upper bound: 0.4741462
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4684839, upper bound: 0.4741482
IS_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4667829, upper bound: 0.4719524
IS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4667829, upper bound: 0.4740922
IS_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4689487, upper bound: 0.4732324
IS_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4689488, upper bound: 0.4741186
IS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4669114, upper bound: 0.4719807
IS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4669114, upper bound: 0.4741205
IS_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4690772, upper bound: 0.4732606
IS_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.54
Output dim: 7, lower bound: -0.4690773, upper bound: 0.4741468

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4594135, -9.2625589, -11.4666367, -9.2544918, -1.3000474, 1.3033333
1: -6.4953718, -4.7301121, -6.5111809, -4.7211776, -1.4002624, 1.4057815
2: -6.2047482, -4.2409124, -6.2158422, -4.2212644, -1.3910742, 1.3831637
3: -5.3218083, -3.7655363, -5.3318672, -3.7543571, -1.0167565, 1.0163851
4: -7.3705215, -5.1752443, -7.3874226, -5.1588631, -1.3039367, 1.2986032
5: -10.4413052, -8.6400137, -10.4744349, -8.6210918, -1.0875330, 1.0984256
6: -17.1130810, -14.7378483, -17.1231804, -14.7199039, -1.3250558, 1.3128229
7: 5.0646706, 6.2428956, 5.0551529, 6.2486897, -0.9501698, 0.9505398
8: -6.4213705, -4.6839457, -6.4305038, -4.6780844, -1.0785933, 1.0804145
9: -5.4233603, -3.8221662, -5.4447055, -3.8113766, -1.2865915, 1.2989697

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667817, upper bound: 0.4653061
time: 3.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667817, upper bound: 0.4661892
time: 6.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -11.5117607, -9.2540216, -11.4666367, -9.2544918, -1.3317490, 1.3136127
1: -6.5167742, -4.6787152, -6.5111809, -4.7211776, -1.4243846, 1.4328701
2: -6.2310362, -4.2368212, -6.2158422, -4.2212644, -1.4155657, 1.3880706
3: -5.3480892, -3.6998549, -5.3318672, -3.7543571, -1.0449345, 1.0288105
4: -7.4189119, -5.1633940, -7.3874226, -5.1588631, -1.3270226, 1.3119560
5: -10.4507246, -8.6140509, -10.4744349, -8.6210918, -1.0958862, 1.1148449
6: -17.1623936, -14.7202511, -17.1231804, -14.7199039, -1.3497677, 1.3319722
7: 5.0404816, 6.2498274, 5.0551529, 6.2486897, -0.9665174, 0.9583175
8: -6.4389415, -4.6387072, -6.4305038, -4.6780844, -1.0968339, 1.1011726
9: -5.4668112, -3.8064463, -5.4447055, -3.8113766, -1.3129308, 1.3154707

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667817, upper bound: 0.4674444
time: 3.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4667817, upper bound: 0.4683275
time: 4.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -11.4596310, -9.2606821, -11.4888363, -9.2553968, -1.3042052, 1.2999722
1: -6.4993114, -4.7371984, -6.4971142, -4.6935558, -1.4068339, 1.3952017
2: -6.1952658, -4.2418284, -6.2157345, -4.2665310, -1.3432384, 1.3792500
3: -5.3282676, -3.7648613, -5.3330464, -3.7062044, -1.0149519, 1.0176895
4: -7.3712664, -5.1767035, -7.4033442, -5.1706996, -1.2947407, 1.2807571
5: -10.4435987, -8.6454639, -10.4537582, -8.6120996, -1.0809665, 1.0713954
6: -17.1102028, -14.7297430, -17.1595173, -14.7191916, -1.3207388, 1.3270642
7: 5.0658841, 6.2459078, 5.0412216, 6.2397404, -0.9386535, 0.9598956
8: -6.4285078, -4.6857018, -6.4250917, -4.6425571, -1.0903008, 1.0745630
9: -5.4255776, -3.8382788, -5.4463339, -3.8491344, -1.2491286, 1.2549045

Time for backsubstitution: 12.62 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=0.9791851043701172
rel_dist={7: [-0.4741558180834273, 0.47415573392575094]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 577

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111642, upper bound: 0.4073550
time: 7.47 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111771, upper bound: 0.4111762
time: 5.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.16 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.16
Output dim: 7, lower bound: -0.4111642, upper bound: 0.4073550
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.16
Output dim: 7, lower bound: -0.4111771, upper bound: 0.4111762

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

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6153

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106953, upper bound: 0.4073545
time: 6.47 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111635, upper bound: 0.4073543
time: 7.96 seconds

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

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6153
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107144, upper bound: 0.4111753
time: 5.71 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111763, upper bound: 0.4111756
time: 4.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.48 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.48
Output dim: 7, lower bound: -0.4106953, upper bound: 0.4073545
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.48
Output dim: 7, lower bound: -0.4111635, upper bound: 0.4073543
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.48
Output dim: 7, lower bound: -0.4107144, upper bound: 0.4111753
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.48
Output dim: 7, lower bound: -0.4111763, upper bound: 0.4111756

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -11.4700127, -9.2492113, -11.4693699, -9.2561798, -1.2175841, 1.2206178
1: -6.5180459, -4.7181172, -6.5046539, -4.7264881, -1.3509953, 1.3469353
2: -6.2163858, -4.2204247, -6.2137632, -4.2391000, -1.3175964, 1.3329327
3: -5.3477955, -3.7533917, -5.3398662, -3.7620063, -0.9721076, 0.9739192
4: -7.3883152, -5.1493273, -7.3760376, -5.1654005, -1.2269568, 1.2343298
5: -10.4782476, -8.6216927, -10.4519997, -8.6363678, -1.0333598, 1.0258524
6: -17.1238365, -14.7092228, -17.1189365, -14.7262077, -1.2277884, 1.2432975
7: 5.0529294, 6.2531347, 5.0612583, 6.2508945, -0.9251578, 0.9217434
8: -6.4440889, -4.6775174, -6.4401841, -4.6820278, -1.0225391, 1.0259931
9: -5.4493985, -3.8036063, -5.4287071, -3.8082366, -1.2702811, 1.2520525

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4068796, upper bound: 0.4073543
time: 4.65 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4068796, upper bound: 0.4073549
time: 5.83 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -11.4713354, -9.2479620, -11.4776869, -9.2470236, -1.2237599, 1.2399457
1: -6.5234823, -4.7154155, -6.5250540, -4.7153172, -1.3681052, 1.3716323
2: -6.2258916, -4.2193704, -6.2329106, -4.2185659, -1.3423719, 1.3480005
3: -5.3554697, -3.7503879, -5.3563685, -3.7483115, -0.9926379, 0.9915922
4: -7.4019389, -5.1485796, -7.4044151, -5.1484013, -1.2523274, 1.2439218
5: -10.4817085, -8.6021137, -10.4879999, -8.6009874, -1.0451477, 1.0611138
6: -17.1340294, -14.7069778, -17.1376953, -14.7063665, -1.2491412, 1.2554868
7: 5.0498323, 6.2543139, 5.0491014, 6.2576599, -0.9371512, 0.9329820
8: -6.4465551, -4.6751704, -6.4514027, -4.6742067, -1.0467396, 1.0309074
9: -5.4514360, -3.7926555, -5.4517450, -3.7882311, -1.2746146, 1.2890704

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4073551, upper bound: 0.4073545
time: 4.45 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4073578, upper bound: 0.4073543
time: 4.85 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.4831848, -9.1918736, -11.4736671, -9.2555742, -1.2299709, 1.2498847
1: -6.5227098, -4.7112727, -6.5057316, -4.7264228, -1.3634040, 1.3556440
2: -6.2297993, -4.1536026, -6.2185102, -4.2385759, -1.3297176, 1.3668720
3: -5.3522635, -3.7408834, -5.3404808, -3.7606781, -0.9776964, 0.9859778
4: -7.3942595, -5.1323938, -7.3777175, -5.1652799, -1.2332578, 1.2527308
5: -10.4891768, -8.5778046, -10.4562492, -8.6355906, -1.0434458, 1.0403371
6: -17.1317558, -14.6838322, -17.1214237, -14.7258148, -1.2350965, 1.2594943
7: 5.0231643, 6.2589197, 5.0607848, 6.2531557, -0.9440441, 0.9263852
8: -6.4532127, -4.6394463, -6.4434624, -4.6814065, -1.0300519, 1.0519704
9: -5.4950275, -3.7953107, -5.4289122, -3.8052413, -1.3050680, 1.2586589

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4106147, upper bound: 0.4095454
time: 4.59 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107123, upper bound: 0.4111765
time: 3.67 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.4845037, -9.1906233, -11.4819851, -9.2464180, -1.2361436, 1.2616488
1: -6.5281134, -4.7085710, -6.5261364, -4.7152524, -1.3805537, 1.3803480
2: -6.2393031, -4.1525435, -6.2376580, -4.2180419, -1.3545229, 1.3756438
3: -5.3599348, -3.7378793, -5.3569808, -3.7469816, -0.9982204, 1.0001719
4: -7.4078856, -5.1316462, -7.4060955, -5.1482806, -1.2571082, 1.2562768
5: -10.4926338, -8.5582209, -10.4922485, -8.6002083, -1.0552402, 1.0742671
6: -17.1419506, -14.6815920, -17.1401844, -14.7059746, -1.2564597, 1.2631793
7: 5.0200706, 6.2601004, 5.0486245, 6.2599216, -0.9501817, 0.9376291
8: -6.4556742, -4.6370993, -6.4546819, -4.6735849, -1.0542805, 1.0520023
9: -5.4970665, -3.7843695, -5.4519506, -3.7852380, -1.3051920, 1.2956553

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4095465, upper bound: 0.4110879
time: 5.46 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4111743, upper bound: 0.4111735
time: 4.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.65 seconds
IS_A1_B1_B1, status: Status.VERIFIED, split count: 3, time: 24.65
Output dim: 7, lower bound: -0.4068796, upper bound: 0.4073543
IS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 24.65
Output dim: 7, lower bound: -0.4068796, upper bound: 0.4073549
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 24.65
Output dim: 7, lower bound: -0.4073551, upper bound: 0.4073545
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 24.65
Output dim: 7, lower bound: -0.4073578, upper bound: 0.4073543
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.65
Output dim: 7, lower bound: -0.4106147, upper bound: 0.4095454
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.65
Output dim: 7, lower bound: -0.4107123, upper bound: 0.4111765
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 24.65
Output dim: 7, lower bound: -0.4095465, upper bound: 0.4110879
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 24.65
Output dim: 7, lower bound: -0.4111743, upper bound: 0.4111735

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.4796019, -9.1973448, -11.4725256, -9.2573175, -1.2229714, 1.2429371
1: -6.5149655, -4.7147517, -6.5032353, -4.7275262, -1.3525560, 1.3493237
2: -6.2277865, -4.1546082, -6.2178736, -4.2389121, -1.3276253, 1.3645010
3: -5.3351393, -3.7423403, -5.3349781, -3.7611125, -0.9606268, 0.9793992
4: -7.3912382, -5.1420469, -7.3767390, -5.1683803, -1.2251930, 1.2417065
5: -10.4847956, -8.5802736, -10.4548264, -8.6363659, -1.0378890, 1.0362638
6: -17.1295204, -14.6948872, -17.1207390, -14.7293673, -1.2287512, 1.2469721
7: 5.0258737, 6.2542868, 5.0616407, 6.2516632, -0.9390469, 0.9195890
8: -6.4392371, -4.6403990, -6.4389639, -4.6816897, -1.0157645, 1.0451460
9: -5.4900150, -3.8047998, -5.4273710, -3.8082914, -1.2963309, 1.2474954

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4067797, upper bound: 0.4095316
time: 5.53 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4067797, upper bound: 0.4095457
time: 4.30 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.5325422, -9.1887999, -11.4736576, -9.2555923, -1.2549384, 1.2535622
1: -6.5362625, -4.6633229, -6.5057154, -4.7264276, -1.3835368, 1.3744327
2: -6.2542148, -4.1504197, -6.2185078, -4.2385759, -1.3523417, 1.3700089
3: -5.3615475, -3.6758113, -5.3404627, -3.7606802, -0.9848731, 0.9917498
4: -7.4406195, -5.1301770, -7.3777113, -5.1652951, -1.2494082, 1.2536765
5: -10.4942284, -8.5539351, -10.4562435, -8.6355953, -1.0464334, 1.0459019
6: -17.1801224, -14.6772242, -17.1214218, -14.7258453, -1.2565203, 1.2654530
7: 5.0013719, 6.2612524, 5.0607901, 6.2531500, -0.9504640, 0.9281309
8: -6.4568567, -4.5923481, -6.4434342, -4.6814079, -1.0331144, 1.0578207
9: -5.5348005, -3.7889874, -5.4289036, -3.8052518, -1.3117876, 1.2635839

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068773, upper bound: 0.4111637
time: 3.66 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068773, upper bound: 0.4111737
time: 4.83 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -11.4833755, -9.1923790, -11.4784203, -9.2519007, -1.2291167, 1.2545284
1: -6.5256057, -4.7096558, -6.5183887, -4.7187200, -1.3742337, 1.3695588
2: -6.2386556, -4.1528807, -6.2356453, -4.2190552, -1.3521240, 1.3735473
3: -5.3544502, -3.7383151, -5.3398762, -3.7484393, -0.9923127, 0.9831319
4: -7.4069118, -5.1347418, -7.4030838, -5.1579318, -1.2460964, 1.2478400
5: -10.4912176, -8.5589809, -10.4878607, -8.6026649, -1.0511942, 1.0687060
6: -17.1412468, -14.6851473, -17.1379356, -14.7170076, -1.2439036, 1.2564117
7: 5.0209436, 6.2586145, 5.0513639, 6.2552900, -0.9433329, 0.9327786
8: -6.4511728, -4.6373796, -6.4407043, -4.6745367, -1.0474541, 1.0376164
9: -5.4955397, -3.7874129, -5.4469395, -3.7947080, -1.2939112, 1.2871680

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057234, upper bound: 0.4110746
time: 4.83 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057234, upper bound: 0.4110877
time: 6.68 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -11.4844999, -9.1906376, -11.5313511, -9.2434206, -1.2407255, 1.2782567
1: -6.5280948, -4.7085757, -6.5396004, -4.6673570, -1.3957109, 1.4001307
2: -6.2392993, -4.1525450, -6.2620645, -4.2148485, -1.3576415, 1.3902037
3: -5.3599167, -3.7378809, -5.3662882, -3.6819303, -1.0059078, 1.0075188
4: -7.4078789, -5.1316605, -7.4524212, -5.1460567, -1.2580402, 1.2634715
5: -10.4926291, -8.5582266, -10.4973764, -8.5762997, -1.0607860, 1.0771751
6: -17.1419487, -14.6816196, -17.1883965, -14.6994209, -1.2623568, 1.2757943
7: 5.0200748, 6.2600956, 5.0268316, 6.2623005, -0.9519331, 0.9455714
8: -6.4556494, -4.6371012, -6.4583130, -4.6264830, -1.0601275, 1.0550431
9: -5.4970565, -3.7843816, -5.4917426, -3.7789168, -1.3098614, 1.3026171

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 577

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073528, upper bound: 0.4111607
time: 4.31 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073528, upper bound: 0.4111746
time: 6.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.96 seconds
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.96
Output dim: 7, lower bound: -0.4067797, upper bound: 0.4095316
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.96
Output dim: 7, lower bound: -0.4067797, upper bound: 0.4095457
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.96
Output dim: 7, lower bound: -0.4068773, upper bound: 0.4111637
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.96
Output dim: 7, lower bound: -0.4068773, upper bound: 0.4111737
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.96
Output dim: 7, lower bound: -0.4057234, upper bound: 0.4110746
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.96
Output dim: 7, lower bound: -0.4057234, upper bound: 0.4110877
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.96
Output dim: 7, lower bound: -0.4073528, upper bound: 0.4111607
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.96
Output dim: 7, lower bound: -0.4073528, upper bound: 0.4111746

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.4796009, -9.1976786, -11.4618721, -9.2588625, -1.2239780, 1.2317687
1: -6.5149336, -4.7147512, -6.5005884, -4.7276912, -1.3431814, 1.3459740
2: -6.2277861, -4.1550078, -6.2061019, -4.2402387, -1.3284645, 1.3521601
3: -5.3351393, -3.7424302, -5.3334570, -3.7645185, -0.9588394, 0.9769913
4: -7.3912373, -5.1420469, -7.3725758, -5.1686783, -1.2238264, 1.2383542
5: -10.4847965, -8.5806847, -10.4442854, -8.6382780, -1.0385869, 1.0252844
6: -17.1295204, -14.6950188, -17.1145782, -14.7303648, -1.2294745, 1.2404951
7: 5.0259781, 6.2542868, 5.0628414, 6.2460556, -0.9331801, 0.9205773
8: -6.4392385, -4.6405711, -6.4308395, -4.6832752, -1.0172436, 1.0367315
9: -5.4898262, -3.8048015, -5.4268603, -3.8157206, -1.2887099, 1.2490599

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4061173, upper bound: 0.4095294
time: 4.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4067777, upper bound: 0.4095295
time: 4.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.4796019, -9.1973448, -11.4750481, -9.2015152, -1.2476745, 1.2444060
1: -6.5149655, -4.7147517, -6.5052700, -4.7208471, -1.3549838, 1.3545482
2: -6.2277865, -4.1546082, -6.2194996, -4.1734037, -1.3568022, 1.3646333
3: -5.3351393, -3.7423403, -5.3379178, -3.7520099, -0.9696505, 0.9826603
4: -7.3912382, -5.1420469, -7.3785119, -5.1517391, -1.2367845, 1.2392733
5: -10.4847956, -8.5802736, -10.4552135, -8.5943880, -1.0533085, 1.0356692
6: -17.1295204, -14.6948872, -17.1225033, -14.7049694, -1.2416177, 1.2478421
7: 5.0258737, 6.2542868, 5.0330553, 6.2518420, -0.9378169, 0.9359140
8: -6.4392371, -4.6403990, -6.4399662, -4.6452017, -1.0383339, 1.0443964
9: -5.4900150, -3.8047998, -5.4724960, -3.8074448, -1.2953296, 1.2775222

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4061146, upper bound: 0.4095439
time: 5.93 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4067777, upper bound: 0.4095445
time: 4.22 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.5325384, -9.1891346, -11.4630089, -9.2571354, -1.2511997, 1.2423947
1: -6.5362306, -4.6633224, -6.5030651, -4.7265930, -1.3785486, 1.3710760
2: -6.2542143, -4.1508198, -6.2067347, -4.2399049, -1.3479629, 1.3576676
3: -5.3615460, -3.6759021, -5.3389425, -3.7640862, -0.9830849, 0.9893419
4: -7.4406190, -5.1301775, -7.3735452, -5.1655908, -1.2489278, 1.2503238
5: -10.4942284, -8.5543461, -10.4457016, -8.6375065, -1.0471315, 1.0349233
6: -17.1801224, -14.6773567, -17.1152592, -14.7268457, -1.2546959, 1.2589755
7: 5.0014772, 6.2612524, 5.0619898, 6.2475419, -0.9445968, 0.9291203
8: -6.4568548, -4.5925207, -6.4353108, -4.6829948, -1.0345943, 1.0494064
9: -5.5346127, -3.7889888, -5.4283929, -3.8126802, -1.3041663, 1.2651513

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4062122, upper bound: 0.4111619
time: 3.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068753, upper bound: 0.4111585
time: 3.97 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.5325422, -9.1887999, -11.4761810, -9.1997890, -1.2714446, 1.2550273
1: -6.5362625, -4.6633229, -6.5077496, -4.7197490, -1.3859806, 1.3774973
2: -6.2542148, -4.1504197, -6.2201371, -4.1730719, -1.3735206, 1.3701441
3: -5.3615475, -3.6758113, -5.3433990, -3.7515769, -0.9940218, 0.9950092
4: -7.4406195, -5.1301770, -7.3794851, -5.1486564, -1.2559354, 1.2512467
5: -10.4942284, -8.5539351, -10.4566317, -8.5936165, -1.0619247, 1.0453076
6: -17.1801224, -14.6772242, -17.1231842, -14.7014475, -1.2610087, 1.2663231
7: 5.0013719, 6.2612524, 5.0322061, 6.2533274, -0.9492333, 0.9444826
8: -6.4568567, -4.5923481, -6.4444380, -4.6449203, -1.0557766, 1.0570706
9: -5.5348005, -3.7889874, -5.4740248, -3.8044043, -1.3107860, 1.2935545

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4585
type: B, layer: 1, pos: 4585
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4585

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4062148, upper bound: 0.4111747
time: 4.22 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4068753, upper bound: 0.4111729
time: 3.96 seconds

## BFS IS instance: IS_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -11.4833736, -9.1927118, -11.4677677, -9.2534475, -1.2300134, 1.2433602
1: -6.5255733, -4.7096558, -6.5157337, -4.7188835, -1.3648405, 1.3662088
2: -6.2386575, -4.1532807, -6.2238746, -4.2203856, -1.3477371, 1.3612059
3: -5.3544502, -3.7384057, -5.3383589, -3.7518444, -0.9905241, 0.9807273
4: -7.4069118, -5.1347423, -7.3989215, -5.1582298, -1.2456167, 1.2444869
5: -10.4912167, -8.5593929, -10.4773207, -8.6045771, -1.0470278, 1.0577264
6: -17.1412468, -14.6852779, -17.1317749, -14.7180109, -1.2420797, 1.2499350
7: 5.0210471, 6.2586145, 5.0525742, 6.2496819, -0.9374654, 0.9326053
8: -6.4511724, -4.6375527, -6.4325743, -4.6761236, -1.0452540, 1.0292032
9: -5.4953518, -3.7874143, -5.4464297, -3.8021324, -1.2862878, 1.2853026

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_B1_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057216, upper bound: 0.4104092
time: 7.91 seconds

## Relational analysis of IS_A2_B2_B1_B1_B2

### Relational analysis result of IS_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057216, upper bound: 0.4110731
time: 4.00 seconds

## BFS IS instance: IS_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -11.4833755, -9.1923790, -11.4809475, -9.1961040, -1.2502484, 1.2560002
1: -6.5256057, -4.7096558, -6.5203629, -4.7120352, -1.3766887, 1.3747811
2: -6.2386556, -4.1528807, -6.2372808, -4.1535563, -1.3732855, 1.3736666
3: -5.3544502, -3.7383151, -5.3428335, -3.7393367, -0.9962298, 0.9864068
4: -7.4069118, -5.1347418, -7.4048653, -5.1412950, -1.2526231, 1.2491896
5: -10.4912176, -8.5589809, -10.4882479, -8.5606852, -1.0591249, 1.0681078
6: -17.1412468, -14.6851473, -17.1396980, -14.6926422, -1.2483885, 1.2572820
7: 5.0209436, 6.2586145, 5.0228057, 6.2554684, -0.9421020, 0.9434829
8: -6.4511728, -4.6373796, -6.4416914, -4.6380496, -1.0570579, 1.0368719
9: -5.4955397, -3.7874129, -5.4920678, -3.7938507, -1.2928967, 1.3038316

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057216, upper bound: 0.4104226
time: 7.06 seconds

## Relational analysis of IS_A2_B2_B1_B2_B2

### Relational analysis result of IS_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057216, upper bound: 0.4110865
time: 5.87 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -11.4844971, -9.1909714, -11.5207014, -9.2449646, -1.2405581, 1.2670915
1: -6.5280638, -4.7085757, -6.5369501, -4.6675215, -1.3927784, 1.3967897
2: -6.2392998, -4.1529455, -6.2502966, -4.2161775, -1.3532529, 1.3778634
3: -5.3599167, -3.7379723, -5.3647747, -3.6853366, -1.0035259, 1.0051175
4: -7.4078770, -5.1316609, -7.4482617, -5.1463537, -1.2575610, 1.2601198
5: -10.4926291, -8.5586395, -10.4868355, -8.5782175, -1.0566196, 1.0661957
6: -17.1419487, -14.6817522, -17.1822376, -14.7004261, -1.2605360, 1.2693177
7: 5.0201793, 6.2600956, 5.0280409, 6.2566938, -0.9460666, 0.9439585
8: -6.4556494, -4.6372724, -6.4501858, -4.6280680, -1.0579276, 1.0466300
9: -5.4968700, -3.7843823, -5.4912310, -3.7863381, -1.3022370, 1.3007526

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073511, upper bound: 0.4104949
time: 4.00 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073511, upper bound: 0.4111589
time: 4.53 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -11.4844999, -9.1906376, -11.5338650, -9.1876268, -1.2607751, 1.2797306
1: -6.5280948, -4.7085757, -6.5415058, -4.6606822, -1.3981624, 1.4032036
2: -6.2392993, -4.1525450, -6.2637081, -4.1493473, -1.3788004, 1.3902994
3: -5.3599167, -3.7378809, -5.3692098, -3.6728258, -1.0085576, 1.0107524
4: -7.4078789, -5.1316605, -7.4542084, -5.1294117, -1.2645714, 1.2648306
5: -10.4926291, -8.5582266, -10.4977627, -8.5343342, -1.0687211, 1.0765779
6: -17.1419487, -14.6816196, -17.1901588, -14.6750231, -1.2668819, 1.2766647
7: 5.0200748, 6.2600956, 4.9982834, 6.2624803, -0.9507027, 0.9548347
8: -6.4556494, -4.6371012, -6.4593039, -4.5899992, -1.0697331, 1.0542957
9: -5.4970565, -3.7843816, -5.5368481, -3.7780702, -1.3088348, 1.3192620

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4585
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4585

## Relational analysis of IS_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073511, upper bound: 0.4105115
time: 3.88 seconds

## Relational analysis of IS_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4073511, upper bound: 0.4111626
time: 3.92 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.59 seconds
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4061173, upper bound: 0.4095294
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4067777, upper bound: 0.4095295
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4061146, upper bound: 0.4095439
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4067777, upper bound: 0.4095445
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4062122, upper bound: 0.4111619
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4068753, upper bound: 0.4111585
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4062148, upper bound: 0.4111747
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4068753, upper bound: 0.4111729
IS_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4057216, upper bound: 0.4104092
IS_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4057216, upper bound: 0.4110731
IS_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4057216, upper bound: 0.4104226
IS_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4057216, upper bound: 0.4110865
IS_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4073511, upper bound: 0.4104949
IS_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4073511, upper bound: 0.4111589
IS_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4073511, upper bound: 0.4105115
IS_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.4073511, upper bound: 0.4111626

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4489069, -9.2070894, -11.4578352, -9.2630596, -1.1837609, 1.2192338
1: -6.4793611, -4.7392015, -6.4961352, -4.7402067, -1.2930999, 1.3141556
2: -6.2008371, -4.2044702, -6.1925592, -4.2425342, -1.2973371, 1.2933511
3: -5.3098478, -3.7610481, -5.3208504, -3.7654526, -0.9329236, 0.9423462
4: -7.3575420, -5.1657586, -7.3698568, -5.1818042, -1.1744721, 1.2116019
5: -10.4549389, -8.5982513, -10.4417849, -8.6476831, -0.9994502, 1.0026338
6: -17.1142883, -14.7119141, -17.1086388, -14.7337999, -1.2060733, 1.2146184
7: 5.0367312, 6.2385645, 5.0674410, 6.2440987, -0.9193935, 0.8980148
8: -6.4163117, -4.6539817, -6.4228230, -4.6864977, -0.9914775, 1.0109179
9: -5.4463377, -3.8584445, -5.4235067, -3.8459725, -1.2025471, 1.1899524

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 4585

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4046036, upper bound: 0.4095293
time: 4.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4046036, upper bound: 0.4095298
time: 4.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -11.4795933, -9.1976881, -11.4618702, -9.2588673, -1.2174077, 1.2244849
1: -6.5149264, -4.7147751, -6.5005846, -4.7277012, -1.3386505, 1.3277793
2: -6.2277584, -4.1550126, -6.2060890, -4.2402401, -1.3038154, 1.3334540
3: -5.3351212, -3.7424328, -5.3334503, -3.7645183, -0.9444106, 0.9679769
4: -7.3912339, -5.1420727, -7.3725719, -5.1686921, -1.2155027, 1.2086174
5: -10.4847918, -8.5806961, -10.4442816, -8.6382809, -1.0265882, 1.0025392
6: -17.1295071, -14.6950254, -17.1145744, -14.7303677, -1.2286716, 1.2262700
7: 5.0259848, 6.2542820, 5.0628443, 6.2460537, -0.9272280, 0.9180678
8: -6.4392223, -4.6405735, -6.4308319, -4.6832771, -1.0161953, 1.0296263
9: -5.4898214, -3.8048396, -5.4268575, -3.8157301, -1.2567596, 1.1913033

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4052666, upper bound: 0.4095295
time: 4.26 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4052666, upper bound: 0.4095300
time: 3.98 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.4489069, -9.2067585, -11.4710188, -9.2057190, -1.2070224, 1.2318815
1: -6.4793925, -4.7392025, -6.5008421, -4.7333641, -1.3049212, 1.3227258
2: -6.2008400, -4.2040701, -6.2059593, -4.1757069, -1.3256075, 1.3058121
3: -5.3098497, -3.7609577, -5.3253160, -3.7529438, -0.9436986, 0.9480152
4: -7.3575430, -5.1657581, -7.3757982, -5.1648645, -1.1874356, 1.2123826
5: -10.4549379, -8.5978384, -10.4527149, -8.6037989, -1.0127938, 1.0130196
6: -17.1142883, -14.7117805, -17.1165600, -14.7084026, -1.2183952, 1.2219660
7: 5.0366268, 6.2385645, 5.0376654, 6.2498856, -0.9240301, 0.9132979
8: -6.4163141, -4.6538100, -6.4319496, -4.6484241, -1.0129240, 1.0185810
9: -5.4465265, -3.8584428, -5.4691453, -3.8376844, -1.2091751, 1.2182682

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 4585

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4047620, upper bound: 0.4095437
time: 4.49 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4047646, upper bound: 0.4095474
time: 4.02 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.4795952, -9.1973543, -11.4750452, -9.2015209, -1.2376537, 1.2371279
1: -6.5149593, -4.7147756, -6.5052671, -4.7208557, -1.3440275, 1.3343743
2: -6.2277575, -4.1546116, -6.2194896, -4.1734052, -1.3293796, 1.3459281
3: -5.3351212, -3.7423422, -5.3379107, -3.7520099, -0.9535232, 0.9736460
4: -7.3912344, -5.1420736, -7.3785095, -5.1517544, -1.2225099, 1.2133249
5: -10.4847898, -8.5802832, -10.4552107, -8.5943918, -1.0386820, 1.0129240
6: -17.1295071, -14.6948938, -17.1224957, -14.7049751, -1.2350225, 1.2336173
7: 5.0258794, 6.2542820, 5.0330582, 6.2518411, -0.9318647, 0.9289824
8: -6.4392209, -4.6403990, -6.4399576, -4.6452026, -1.0329652, 1.0372910
9: -5.4900084, -3.8048375, -5.4724936, -3.8074546, -1.2633796, 1.2138863

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4054250, upper bound: 0.4095442
time: 3.95 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4054250, upper bound: 0.4095474
time: 3.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.5016460, -9.1985636, -11.4589901, -9.2613297, -1.2100089, 1.2298834
1: -6.5008097, -4.6872039, -6.4986095, -4.7391138, -1.3284304, 1.3396490
2: -6.2276549, -4.2002993, -6.1931915, -4.2422032, -1.3169656, 1.2987256
3: -5.3362865, -3.6945410, -5.3263359, -3.7650161, -0.9572654, 0.9546134
4: -7.4067941, -5.1538763, -7.3708296, -5.1787224, -1.1962945, 1.2236124
5: -10.4641085, -8.5718527, -10.4431982, -8.6469078, -1.0077415, 1.0123267
6: -17.1654472, -14.6942749, -17.1093121, -14.7302790, -1.2311139, 1.2330593
7: 5.0122032, 6.2453399, 5.0665894, 6.2455869, -0.9308476, 0.9063523
8: -6.4339085, -4.6059399, -6.4272866, -4.6862135, -1.0088401, 1.0235821
9: -5.4910383, -3.8425846, -5.4250550, -3.8429322, -1.2177193, 1.2060306

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 4585

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4062148, upper bound: 0.4106933
time: 4.11 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4062122, upper bound: 0.4111619
time: 3.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.5325346, -9.1891403, -11.4630070, -9.2571421, -1.2407825, 1.2351360
1: -6.5362258, -4.6633453, -6.5030627, -4.7266021, -1.3636417, 1.3455768
2: -6.2541862, -4.1508250, -6.2067223, -4.2399073, -1.3207633, 1.3389454
3: -5.3615284, -3.6759036, -5.3389335, -3.7640872, -0.9686568, 0.9802722
4: -7.4406137, -5.1302042, -7.3735447, -5.1656065, -1.2310421, 1.2206852
5: -10.4942236, -8.5543594, -10.4456997, -8.6375093, -1.0350637, 1.0122783
6: -17.1801109, -14.6773615, -17.1152554, -14.7268467, -1.2478268, 1.2448255
7: 5.0014853, 6.2612486, 5.0619936, 6.2475414, -0.9386752, 0.9265407
8: -6.4568391, -4.5925226, -6.4353051, -4.6829953, -1.0335453, 1.0422935
9: -5.5346069, -3.7890263, -5.4283886, -3.8126919, -1.2720942, 1.2074351

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4052666, upper bound: 0.4110728
time: 4.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4052666, upper bound: 0.4111590
time: 4.09 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.5016479, -9.1982327, -11.4721718, -9.2039909, -1.2302544, 1.2425270
1: -6.5008411, -4.6872048, -6.5033178, -4.7322731, -1.3358510, 1.3460914
2: -6.2276549, -4.1999002, -6.2065940, -4.1753745, -1.3425267, 1.3111887
3: -5.3362875, -3.6944513, -5.3307962, -3.7525070, -0.9681711, 0.9602799
4: -7.4067955, -5.1538773, -7.3767748, -5.1617851, -1.2033033, 1.2244635
5: -10.4641085, -8.5714426, -10.4541273, -8.6030264, -1.0211535, 1.0227126
6: -17.1654472, -14.6941414, -17.1172352, -14.7048779, -1.2374530, 1.2404062
7: 5.0120964, 6.2453399, 5.0368156, 6.2513757, -0.9354842, 0.9216614
8: -6.4339075, -4.6057673, -6.4364166, -4.6481447, -1.0303848, 1.0312455
9: -5.4912271, -3.8425832, -5.4706888, -3.8346431, -1.2243471, 1.2342808

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 4585

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6153

## Relational analysis of IS_A2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4063704, upper bound: 0.4107089
time: 4.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4063704, upper bound: 0.4111714
time: 4.32 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.5325365, -9.1888094, -11.4761829, -9.1997929, -1.2610271, 1.2477739
1: -6.5362558, -4.6633444, -6.5077486, -4.7197595, -1.3690219, 1.3520141
2: -6.2541857, -4.1504245, -6.2201242, -4.1730728, -1.3463249, 1.3514211
3: -5.3615294, -3.6758137, -5.3433924, -3.7515776, -0.9778984, 0.9859396
4: -7.4406157, -5.1302032, -7.3794823, -5.1486692, -1.2380493, 1.2253932
5: -10.4942245, -8.5539484, -10.4566307, -8.5936203, -1.0471575, 1.0226636
6: -17.1801109, -14.6772308, -17.1231766, -14.7014484, -1.2541759, 1.2521724
7: 5.0013800, 6.2612486, 5.0322094, 6.2533278, -0.9433119, 0.9374545
8: -6.4568396, -4.5923510, -6.4444284, -4.6449213, -1.0504076, 1.0499580
9: -5.5347948, -3.7890246, -5.4740214, -3.8044155, -1.2787147, 1.2299516

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 6153
type: B, layer: 1, pos: 4585

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6183

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4054250, upper bound: 0.4110863
time: 4.23 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4054250, upper bound: 0.4111738
time: 3.75 seconds

## BFS IS instance: IS_A2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -11.4793653, -9.1967812, -11.4370699, -9.2629652, -1.2170458, 1.2031308
1: -6.5212669, -4.7222304, -6.4802313, -4.7434092, -1.3330181, 1.3165076
2: -6.2251048, -4.1555991, -6.1968970, -4.2698936, -1.2893128, 1.3299122
3: -5.3418236, -3.7393966, -5.3130312, -3.7705286, -0.9568676, 0.9546316
4: -7.4039192, -5.1478076, -7.3654852, -5.1819611, -1.2185693, 1.1924061
5: -10.4887295, -8.5687933, -10.4474583, -8.6221428, -1.0241799, 1.0180217
6: -17.1352177, -14.6887140, -17.1164532, -14.7349205, -1.2159200, 1.2265772
7: 5.0254993, 6.2566004, 5.0633984, 6.2338910, -0.9151661, 0.9187641
8: -6.4436731, -4.6407852, -6.4089913, -4.6895685, -1.0199213, 1.0031132
9: -5.4918437, -3.8176908, -5.4028177, -3.8558147, -1.2267356, 1.1993740

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4057216, upper bound: 0.4088656
time: 6.10 seconds

## Relational analysis of IS_A2_B2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057216, upper bound: 0.4104090
time: 6.11 seconds

## BFS IS instance: IS_A2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -11.4833698, -9.1927185, -11.4677649, -9.2534580, -1.2225561, 1.2334607
1: -6.5255694, -4.7096682, -6.5157275, -4.7189069, -1.3464727, 1.3545132
2: -6.2386413, -4.1532826, -6.2238460, -4.2203903, -1.3292291, 1.3331031
3: -5.3544393, -3.7384067, -5.3383389, -3.7518468, -0.9822505, 0.9643786
4: -7.4069080, -5.1347551, -7.3989153, -5.1582546, -1.2156384, 1.2268082
5: -10.4912148, -8.5593996, -10.4773169, -8.6045933, -1.0240326, 1.0431966
6: -17.1412392, -14.6852818, -17.1317635, -14.7180185, -1.2276657, 1.2431520
7: 5.0210524, 6.2586122, 5.0525818, 6.2496796, -0.9305916, 0.9265285
8: -6.4511638, -4.6375551, -6.4325585, -4.6761255, -1.0381494, 1.0237758
9: -5.4953485, -3.7874339, -5.4464231, -3.8021698, -1.2222486, 1.2534074

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057216, upper bound: 0.4095289
time: 4.06 seconds

## Relational analysis of IS_A2_B2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4057216, upper bound: 0.4110731
time: 4.12 seconds

## BFS IS instance: IS_A2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -11.4793644, -9.1964502, -11.4502535, -9.2056293, -1.2375002, 1.2157950
1: -6.5212989, -4.7222304, -6.4848967, -4.7365627, -1.3448122, 1.3250723
2: -6.2251048, -4.1551991, -6.2102909, -4.2030568, -1.3148785, 1.3423622
3: -5.3418255, -3.7393074, -5.3175087, -3.7580218, -0.9618969, 0.9603086
4: -7.4039211, -5.1478071, -7.3714266, -5.1650271, -1.2255745, 1.1971092
5: -10.4887295, -8.5683832, -10.4583883, -8.5782518, -1.0362663, 1.0284053
6: -17.1352177, -14.6885853, -17.1243744, -14.7095366, -1.2222559, 1.2339245
7: 5.0253959, 6.2566004, 5.0336390, 6.2396784, -0.9198027, 0.9296560
8: -6.4436750, -4.6406126, -6.4180942, -4.6514888, -1.0317247, 1.0107566
9: -5.4920311, -3.8176897, -5.4484630, -3.8475480, -1.2333288, 1.2179935

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 467
type: B, layer: 1, pos: 467
type: B, layer: 1, pos: 4602
type: A, layer: 1, pos: 4602
type: A, layer: 1, pos: 4585
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4058767, upper bound: 0.4088802
time: 5.11 seconds

## Relational analysis of IS_A2_B2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058767, upper bound: 0.4104228
time: 4.61 seconds

## BFS IS instance: IS_A2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -11.4833717, -9.1923828, -11.4809370, -9.1961117, -1.2427919, 1.2461088
1: -6.5256023, -4.7096677, -6.5203562, -4.7120585, -1.3548157, 1.3609782
2: -6.2386413, -4.1528835, -6.2372508, -4.1535616, -1.3547847, 1.3455490
3: -5.3544388, -3.7383170, -5.3428144, -3.7393391, -0.9872818, 0.9700587
4: -7.4069090, -5.1347551, -7.4048600, -5.1413226, -1.2226384, 1.2315192
5: -10.4912148, -8.5589876, -10.4882450, -8.5606995, -1.0361230, 1.0535789
6: -17.1412392, -14.6851473, -17.1396866, -14.6926470, -1.2339745, 1.2504990
7: 5.0209470, 6.2586122, 5.0228138, 6.2554660, -0.9352281, 0.9374306
8: -6.4511642, -4.6373820, -6.4416747, -4.6380520, -1.0499539, 1.0314441
9: -5.4955368, -3.7874331, -5.4920611, -3.7938883, -1.2288368, 1.2719450

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6183
type: B, layer: 1, pos: 467
type: A, layer: 1, pos: 467
type: A, layer: 1, pos: 4602
type: B, layer: 1, pos: 4602
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6153
type: A, layer: 1, pos: 4585

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6183

## Relational analysis of IS_A2_B2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058767, upper bound: 0.4095436
time: 4.13 seconds

## Relational analysis of IS_A2_B2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4058767, upper bound: 0.4110864
time: 4.93 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -11.4805021, -9.1950397, -11.4897966, -9.2545128, -1.2278337, 1.2263327
1: -6.5237532, -4.7211556, -6.5015945, -4.6914845, -1.3613443, 1.3471028
2: -6.2257471, -4.1552663, -6.2236762, -4.2657037, -1.2946920, 1.3467791
3: -5.3472910, -3.7389591, -5.3394804, -3.7040429, -0.9691105, 0.9791279
4: -7.4048901, -5.1447315, -7.4146929, -5.1700687, -1.2305601, 1.2078414
5: -10.4901361, -8.5680399, -10.4567165, -8.5957203, -1.0338316, 1.0262696
6: -17.1359138, -14.6851892, -17.1674652, -14.7173595, -1.2343345, 1.2456365
7: 5.0246305, 6.2580848, 5.0388327, 6.2407160, -0.9235620, 0.9301494
8: -6.4481463, -4.6405053, -6.4265842, -4.6415215, -1.0325770, 1.0205619
9: -5.4933753, -3.8146584, -5.4475298, -3.8399737, -1.2426667, 1.2145350

Time for backsubstitution: 14.65 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=0.9420795440673828
rel_dist={7: [-0.411179151475503, 0.41117864435442364]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2411.45 seconds
