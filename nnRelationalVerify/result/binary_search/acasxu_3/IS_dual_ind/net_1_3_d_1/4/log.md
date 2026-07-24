## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 807.3886655422


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953)
1: (-373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194)
2: (-542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741)
3: (-209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672)
4: (-604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148)

## BASE Result
execution time: IAR + LP analysis = 1.60 + 1.89 = 3.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -809.0067386, upper bound: 809.0067386


# Binary Search by BASE starts (time budget: 1196.51 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=1011.34521484375
rel_dist={4: [-809.0067385931752, 809.0067385931752]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=1011.34521484375
rel_dist={4: [-809.0065995903992, 809.0065995903992]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=1011.34521484375
rel_dist={4: [-809.0063734281499, 809.00637342815]}

## Binary search (step 3) starts
Candidate diff: 0.0208333


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0208333, mid=0.0208333, abs_max=1011.34521484375
rel_dist={4: [-809.0059584983813, 809.0059584983815]}

## Binary search (step 4) starts
Candidate diff: 0.0104167


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0104167, mid=0.0104167, abs_max=1011.34521484375
rel_dist={4: [-809.0054003420871, 809.0054003420869]}

## Binary search (step 5) starts
Candidate diff: 0.0052083


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0052083, mid=0.0052083, abs_max=1011.34521484375
rel_dist={4: [-809.005006272113, 809.005006272113]}

## Binary search (step 6) starts
Candidate diff: 0.0026042


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0026042, mid=0.0026042, abs_max=1011.34521484375
rel_dist={4: [-809.0047726399445, 809.0047726399443]}

## Binary search (step 7) starts
Candidate diff: 0.0013021


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0013021, mid=0.0013021, abs_max=1011.34521484375
rel_dist={4: [-809.0045709298558, 809.0045709298556]}

## Binary search (step 8) starts
Candidate diff: 0.0006510


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0006510, mid=0.0006510, abs_max=1011.34521484375
rel_dist={4: [-809.0044539569961, 809.0044539569963]}

## Binary search (step 9) starts
Candidate diff: 0.0003255


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0003255, mid=0.0003255, abs_max=1011.34521484375
rel_dist={4: [-809.0043787668834, 809.0043787668833]}

## Binary search (step 10) starts
Candidate diff: 0.0001628


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0001628, mid=0.0001628, abs_max=1011.34521484375
rel_dist={4: [-809.0043408960324, 809.0043408960323]}

## Binary search (step 11) starts
Candidate diff: 0.0000814


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000814, mid=0.0000814, abs_max=1011.34521484375
rel_dist={4: [-809.0043219310879, 809.0043219310878]}

## Binary search (step 12) starts
Candidate diff: 0.0000407


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000407, mid=0.0000407, abs_max=1011.34521484375
rel_dist={4: [-809.0043123987718, 809.0043123987716]}

## Binary search (step 13) starts
Candidate diff: 0.0000203


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000203, mid=0.0000203, abs_max=1011.34521484375
rel_dist={4: [-809.0043076328072, 809.004307632807]}

## Binary search (step 14) starts
Candidate diff: 0.0000102


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000102, mid=0.0000102, abs_max=1011.34521484375
rel_dist={4: [-809.0043052501968, 809.0043052501967]}

## Binary search (step 15) starts
Candidate diff: 0.0000051


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000051, mid=0.0000051, abs_max=1011.34521484375
rel_dist={4: [-809.0043044480162, 809.0043040595815]}

## Binary search (step 16) starts
Candidate diff: 0.0000025


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000025, mid=0.0000025, abs_max=1011.34521484375
rel_dist={4: [-809.0043036511679, 809.0043034673447]}

## Binary search (step 17) starts
Candidate diff: 0.0000013


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000013, mid=0.0000013, abs_max=1011.34521484375
rel_dist={4: [-809.0043032290516, 809.0043031699583]}

## Binary search (step 18) starts
Candidate diff: 0.0000006


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000006, mid=0.0000006, abs_max=1011.34521484375
rel_dist={4: [-809.004306129907, 809.0043400607263]}

## Binary Search Result
Binary search time: 66.77 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1129.74 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4692061, upper bound: 809.0031754
time: 0.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0039002, upper bound: 809.0039007
time: 0.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.46 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 4, lower bound: -806.4692061, upper bound: 809.0031754
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 4, lower bound: -809.0039002, upper bound: 809.0039007

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -355.7917175, 298.0696411, -465.1231079, 388.6268616, -744.4185181, 763.1926880
1: -284.4114380, 290.6557922, -372.9984131, 376.8692322, -661.2806396, 663.6541748
2: -411.3229675, 317.9904480, -542.1752319, 411.2849731, -822.6077881, 860.1655884
3: -162.9068451, 405.6134033, -209.7512207, 530.4779663, -693.3848267, 615.3645630
4: -458.9186401, 313.6753845, -603.9422607, 407.1642456, -866.0828857, 917.6176758

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4684808, upper bound: 806.4684808
time: 0.61 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4684808, upper bound: 806.4684808
time: 0.69 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -465.2375793, 388.7125854, -843.7695312, 845.8966064
1: -364.9006348, 369.0178833, -373.0900269, 376.9524841, -741.8531494, 742.1078491
2: -530.4962158, 402.6548462, -542.3095093, 411.3765564, -941.8726807, 944.9643555
3: -205.3341675, 519.1251221, -209.7973175, 530.6040649, -735.9381104, 728.9224243
4: -590.9782715, 398.9001770, -604.0916138, 407.2537537, -998.2319336, 1002.9916992

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0031754, upper bound: 806.4692061
time: 0.72 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0031754, upper bound: 806.4692061
time: 0.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.25 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.25
Output dim: 4, lower bound: -806.4684808, upper bound: 806.4684808
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 3.25
Output dim: 4, lower bound: -806.4684808, upper bound: 806.4684808
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 4, lower bound: -809.0031754, upper bound: 806.4692061
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 4, lower bound: -809.0031754, upper bound: 806.4692061

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -355.7917175, 298.0696411, -753.1265869, 736.4506836
1: -364.9006348, 369.0178833, -284.4114380, 290.6557922, -655.5563965, 653.4293213
2: -530.4962158, 402.6548462, -411.3229675, 317.9904480, -848.4865723, 813.9777832
3: -205.3341675, 519.1251221, -162.9068451, 405.6134033, -610.9473877, 682.0319824
4: -590.9782715, 398.9001770, -458.9186401, 313.6753845, -904.6536865, 857.8186646

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4198193, upper bound: 806.4690913
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5808097, upper bound: 806.4690029
time: 0.64 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -455.0569763, 380.6590271, -835.7160034, 835.7160034
1: -364.9006348, 369.0178833, -364.9006348, 369.0178833, -733.9185181, 733.9185181
2: -530.4962158, 402.6548462, -530.4962158, 402.6548462, -933.1510620, 933.1510620
3: -205.3341675, 519.1251221, -205.3341675, 519.1251221, -724.4591675, 724.4591675
4: -590.9782715, 398.9001770, -590.9782715, 398.9001770, -989.8784180, 989.8782959

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0025031, upper bound: 808.5813317
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5808102, upper bound: 806.4690029
time: 1.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.98 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.98
Output dim: 4, lower bound: -806.4198193, upper bound: 806.4690913
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 4, lower bound: -808.5808097, upper bound: 806.4690029
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 4, lower bound: -809.0025031, upper bound: 808.5813317
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 4, lower bound: -808.5808102, upper bound: 806.4690029

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -355.7917175, 298.0696411, -742.9589233, 728.0826416
1: -356.6080017, 360.9292603, -284.4114380, 290.6557922, -647.2637939, 645.3406982
2: -518.1353760, 393.8309326, -411.3229675, 317.9904480, -836.1256714, 805.1538086
3: -200.9437408, 507.3840332, -162.9068451, 405.6134033, -606.5570068, 670.2908936
4: -577.2812500, 390.2192383, -458.9186401, 313.6753845, -890.9566650, 849.1377563

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5802569, upper bound: 806.4687643
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5189473, upper bound: 806.4204536
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5189473, upper bound: 806.4690029
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -455.0569763, 380.6590271, -818.4312744, 822.4982910
1: -350.6974182, 356.2439270, -364.9006348, 369.0178833, -719.7153320, 721.1445312
2: -509.7792053, 388.8671265, -530.4962158, 402.6548462, -912.4340820, 919.3633423
3: -198.0632477, 499.4380493, -205.3341675, 519.1251221, -717.1883545, 704.7721558
4: -568.3402710, 385.4368286, -590.9782715, 398.9001770, -967.2403564, 976.4151001

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -455.0569763, 380.6590271, -825.5484619, 827.3479004
1: -356.6080017, 360.9292603, -364.9006348, 369.0178833, -725.6258545, 725.8298950
2: -518.1353760, 393.8309326, -530.4962158, 402.6548462, -920.7902222, 924.3271484
3: -200.9437408, 507.3840332, -205.3341675, 519.1251221, -720.0687256, 712.7181396
4: -577.2812500, 390.2192383, -590.9782715, 398.9001770, -976.1813965, 981.1973877

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5812554
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5812554
time: 0.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.98 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -808.5189473, upper bound: 806.4204536
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -808.5189473, upper bound: 806.4690029
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5812554
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5812554

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -341.3818359, 288.1004028, -732.9898682, 713.6727295
1: -356.6080017, 360.9292603, -272.6688843, 281.0004883, -637.6085205, 633.5980835
2: -518.1353760, 393.8309326, -394.2149963, 307.6066589, -825.7420654, 788.0458984
3: -200.9437408, 507.3840332, -157.2244263, 389.4662170, -590.4099731, 664.6084595
4: -577.2812500, 390.2192383, -440.2426147, 303.5801392, -880.8613281, 830.4617920

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5189473, upper bound: 806.4202746
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3420043, upper bound: 806.4203017
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -346.6198730, 290.6638184, -735.5532227, 718.9107056
1: -356.6080017, 360.9292603, -276.9523926, 283.5098572, -640.1178589, 637.8816528
2: -518.1353760, 393.8309326, -400.2306824, 310.2190247, -828.3543701, 794.0615845
3: -200.9437408, 507.3840332, -158.6000671, 395.0196533, -595.9631958, 665.9840698
4: -577.2812500, 390.2192383, -446.5895996, 306.0231628, -883.3044434, 836.8088379

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5189473, upper bound: 806.4688252
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3420043, upper bound: 806.4688013
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -437.7723999, 367.4412842, -805.2136230, 805.2136230
1: -350.6974182, 356.2439270, -350.6974182, 356.2439270, -706.9413452, 706.9413452
2: -509.7792053, 388.8671265, -509.7792053, 388.8671265, -898.6461792, 898.6463013
3: -198.0632477, 499.4380493, -198.0632477, 499.4380493, -697.5012817, 697.5012817
4: -568.3402710, 385.4368286, -568.3402710, 385.4368286, -953.7770996, 953.7770996

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8217030, upper bound: 808.4947861
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0028577, upper bound: 808.5811780
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -444.8894653, 372.2908936, -810.0632935, 812.3307495
1: -350.6974182, 356.2439270, -356.6080017, 360.9292603, -711.6267090, 712.8518677
2: -509.7792053, 388.8671265, -518.1353760, 393.8309326, -903.6101074, 907.0024414
3: -198.0632477, 499.4380493, -200.9437408, 507.3840332, -705.4472656, 700.3817139
4: -568.3402710, 385.4368286, -577.2812500, 390.2192383, -958.5594482, 962.7180786

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8217030, upper bound: 808.4947861
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0028577, upper bound: 808.5811780
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -437.7723999, 367.4412842, -812.3307495, 810.0632324
1: -356.6080017, 360.9292603, -350.6974182, 356.2439270, -712.8519287, 711.6267090
2: -518.1353760, 393.8309326, -509.7792053, 388.8671265, -907.0024414, 903.6101074
3: -200.9437408, 507.3840332, -198.0632477, 499.4380493, -700.3817749, 705.4472656
4: -577.2812500, 390.2192383, -568.3402710, 385.4368286, -962.7180786, 958.5593872

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2533344, upper bound: 808.2641590
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811857, upper bound: 808.5812322
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -444.8894653, 372.2908936, -817.1803589, 817.1803589
1: -356.6080017, 360.9292603, -356.6080017, 360.9292603, -717.5372314, 717.5372314
2: -518.1353760, 393.8309326, -518.1353760, 393.8309326, -911.9663086, 911.9663086
3: -200.9437408, 507.3840332, -200.9437408, 507.3840332, -708.3276978, 708.3277588
4: -577.2812500, 390.2192383, -577.2812500, 390.2192383, -967.5004883, 967.5004883

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2533345, upper bound: 808.2641624
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811857, upper bound: 808.5812322
time: 0.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.46 seconds
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 4, lower bound: -808.5189473, upper bound: 806.4202746
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 4, lower bound: -808.3420043, upper bound: 806.4203017
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 4, lower bound: -808.5189473, upper bound: 806.4688252
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 4, lower bound: -808.3420043, upper bound: 806.4688013
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 4, lower bound: -808.8217030, upper bound: 808.4947861
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 4, lower bound: -809.0028577, upper bound: 808.5811780
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 4, lower bound: -808.8217030, upper bound: 808.4947861
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 4, lower bound: -809.0028577, upper bound: 808.5811780
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 4, lower bound: -808.2533344, upper bound: 808.2641590
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 4, lower bound: -808.5811857, upper bound: 808.5812322
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 4, lower bound: -808.2533345, upper bound: 808.2641624
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.46
Output dim: 4, lower bound: -808.5811857, upper bound: 808.5812322

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -380.1496887, 318.2424622, -335.1798096, 283.1971741, -663.3468018, 653.4222412
1: -304.5241394, 308.7597961, -267.6474915, 276.2727051, -580.7968140, 576.4072876
2: -441.9195862, 337.3076477, -386.9039001, 302.4981384, -744.4177246, 724.2114868
3: -171.6442566, 433.2800598, -154.4449768, 382.4222412, -554.0664062, 587.7250366
4: -491.8168945, 333.9521179, -432.1092224, 298.4997864, -790.3166504, 766.0613403

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5180422, upper bound: 806.4199436
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4789148, upper bound: 804.9119819
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2180071
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -436.6898499, 365.1800842, -341.3818359, 288.1004028, -724.7902832, 706.5618286
1: -349.9454956, 354.0716248, -272.6688843, 281.0004883, -630.9459839, 626.7404175
2: -508.4296265, 386.5229187, -394.2149963, 307.6066589, -816.0362549, 780.7379150
3: -197.2324371, 497.9155884, -157.2244263, 389.4662170, -586.6986694, 655.1400146
4: -566.5614014, 382.8739319, -440.2426147, 303.5801392, -870.1414795, 823.1163940

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3413808, upper bound: 806.4194732
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7105309, upper bound: 804.9127837
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2024902, upper bound: 806.2180104
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -380.1496887, 318.2424622, -341.2744141, 286.3883057, -666.5377808, 659.5168457
1: -304.5241394, 308.7597961, -272.6340332, 279.3882446, -583.9123535, 581.3937988
2: -441.9195862, 337.3076477, -393.9496460, 305.7650146, -747.6845703, 731.2573242
3: -171.6442566, 433.2800598, -156.1626587, 388.9434814, -560.5876465, 589.4426880
4: -491.8168945, 333.9521179, -439.5921936, 301.5909424, -793.4078369, 773.5443115

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5180422, upper bound: 806.4686374
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8437352, upper bound: 805.8372850
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2180071
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -436.6898499, 365.1800842, -346.6198730, 290.6638184, -727.3536377, 711.7998657
1: -349.9454956, 354.0716248, -276.9523926, 283.5098572, -633.4552612, 631.0240479
2: -508.4296265, 386.5229187, -400.2306824, 310.2190247, -818.6486816, 786.7535400
3: -197.2324371, 497.9155884, -158.6000671, 395.0196533, -592.2518921, 656.5156250
4: -566.5614014, 382.8739319, -446.5895996, 306.0231628, -872.5845947, 829.4634399

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3413808, upper bound: 806.4679166
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8416100, upper bound: 805.8371398
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2024074, upper bound: 806.2180104
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -394.8950500, 334.1777039, -437.7723999, 367.4412842, -762.3363037, 771.9500732
1: -316.1249084, 323.8029480, -350.6974182, 356.2439270, -672.3688354, 674.5003662
2: -459.4748840, 353.5050354, -509.7792053, 388.8671265, -848.3419800, 863.2839966
3: -179.7761841, 450.4103699, -198.0632477, 499.4380493, -679.2141113, 648.4735718
4: -512.8085938, 350.7399292, -568.3402710, 385.4368286, -898.2454224, 919.0802002

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8211117, upper bound: 808.8201044
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8211117, upper bound: 808.8365388
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -673.0687256, 546.6938477, -431.3099365, 361.6101990, -1034.6788330, 973.6739502
1: -541.0684204, 531.0290527, -345.4494629, 350.6086426, -891.6770020, 872.5023804
2: -782.8427124, 578.2561035, -502.0658569, 382.9005127, -1165.7431641, 1075.9584961
3: -297.9616089, 757.8487549, -195.2224274, 491.7924500, -785.9398193, 953.0711670
4: -870.5672607, 571.6234131, -559.7907104, 379.4403687, -1250.0075684, 1127.2396240

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0020928, upper bound: 808.8218766
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0020928, upper bound: 809.0028246
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -394.8950500, 334.1777039, -444.8894653, 372.2908936, -767.1859131, 779.0671387
1: -316.1249084, 323.8029480, -356.6080017, 360.9292603, -677.0541992, 680.4108276
2: -459.4748840, 353.5050354, -518.1353760, 393.8309326, -853.3057861, 871.6402588
3: -179.7761841, 450.4103699, -200.9437408, 507.3840332, -687.1600952, 651.3540649
4: -512.8085938, 350.7399292, -577.2812500, 390.2192383, -903.0277100, 928.0211792

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8211769, upper bound: 808.2533037
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8211769, upper bound: 808.4947861
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -673.0687256, 546.6938477, -438.5579529, 366.5933533, -1039.6619873, 980.9622803
1: -541.0684204, 531.0290527, -351.4568176, 355.4830627, -896.5513916, 878.5432739
2: -782.8427124, 578.2561035, -510.5514526, 388.0996704, -1170.9422607, 1084.5173340
3: -297.9616089, 757.8487549, -198.1423950, 499.8696899, -794.0597534, 955.9911499
4: -870.5672607, 571.6234131, -568.8773804, 384.4866638, -1255.0539551, 1136.4091797

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0021580, upper bound: 808.2540709
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0021580, upper bound: 808.5811780
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -400.8190918, 338.1311340, -437.7723999, 367.4412842, -768.2603760, 775.9033813
1: -321.0563354, 327.5602722, -350.6974182, 356.2439270, -677.3002319, 678.2576904
2: -466.4544678, 357.4834900, -509.7792053, 388.8671265, -855.3215942, 867.2626953
3: -182.1852417, 457.0790405, -198.0632477, 499.4380493, -681.6231689, 655.1421509
4: -520.2389526, 354.6555786, -568.3402710, 385.4368286, -905.6757812, 922.9957275

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2533061, upper bound: 808.2811271
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2533061, upper bound: 808.2811271
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -683.3941650, 554.1363525, -431.3099365, 361.6101990, -1045.0043945, 981.4479980
1: -549.4193726, 538.1006470, -345.4494629, 350.6086426, -900.0280151, 879.8250732
2: -794.7759399, 585.9496460, -502.0658569, 382.9005127, -1177.6765137, 1083.9321289
3: -302.0893250, 768.9462280, -195.2224274, 491.7924500, -790.2961426, 964.1686401
4: -883.6091919, 579.2677612, -559.7907104, 379.4403687, -1263.0493164, 1135.2003174

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5804209, upper bound: 808.8220174
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5804209, upper bound: 809.0003315
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -400.8190918, 338.1311340, -444.8894653, 372.2908936, -773.1099854, 783.0205688
1: -321.0563354, 327.5602722, -356.6080017, 360.9292603, -681.9855347, 684.1682129
2: -466.4544678, 357.4834900, -518.1353760, 393.8309326, -860.2854004, 875.6188965
3: -182.1852417, 457.0790405, -200.9437408, 507.3840332, -689.5692139, 658.0227051
4: -520.2389526, 354.6555786, -577.2812500, 390.2192383, -910.4581909, 931.9367676

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2533061, upper bound: 808.2532643
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2533061, upper bound: 808.2641590
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -683.3941650, 554.1363525, -438.5579529, 366.5933533, -1049.9874268, 988.7364502
1: -549.4193726, 538.1006470, -351.4568176, 355.4830627, -904.9024658, 885.8659668
2: -794.7759399, 585.9496460, -510.5514526, 388.0996704, -1182.8756104, 1092.4913330
3: -302.0893250, 768.9462280, -198.1423950, 499.8696899, -798.4160156, 967.0885620
4: -883.6091919, 579.2677612, -568.8773804, 384.4866638, -1268.0957031, 1144.3698730

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5804209, upper bound: 808.2542117
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5804209, upper bound: 808.5812322
time: 0.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.30 seconds
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.30
Output dim: 4, lower bound: -806.4789148, upper bound: 804.9119819
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2180071
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -807.7105309, upper bound: 804.9127837
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2024902, upper bound: 806.2180104
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -807.8437352, upper bound: 805.8372850
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2180071
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -807.8416100, upper bound: 805.8371398
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2024074, upper bound: 806.2180104
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.8211117, upper bound: 808.8201044
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.8211117, upper bound: 808.8365388
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -809.0020928, upper bound: 808.8218766
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -809.0020928, upper bound: 809.0028246
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.8211769, upper bound: 808.2533037
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.8211769, upper bound: 808.4947861
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -809.0021580, upper bound: 808.2540709
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -809.0021580, upper bound: 808.5811780
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2533061, upper bound: 808.2811271
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2533061, upper bound: 808.2811271
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.5804209, upper bound: 808.8220174
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.5804209, upper bound: 809.0003315
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2533061, upper bound: 808.2532643
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2533061, upper bound: 808.2641590
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.5804209, upper bound: 808.2542117
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.5804209, upper bound: 808.5812322

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -380.1496887, 318.2424622, -327.1813660, 277.3379822, -657.4875488, 645.4237061
1: -304.5241394, 308.7597961, -261.1712952, 270.5529480, -575.0770874, 569.9310913
2: -441.9195862, 337.3076477, -377.5047913, 296.3498230, -738.2694092, 714.8123169
3: -171.6442566, 433.2800598, -151.1579590, 373.3670654, -545.0111084, 584.4379883
4: -491.8168945, 333.9521179, -421.6633911, 292.4135742, -784.2304688, 755.6154785

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2360876, upper bound: 806.2167483
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1762979, upper bound: 806.2167634
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -433.1940002, 362.6026001, -243.2164001, 214.0600586, -647.2539062, 605.8189697
1: -347.1195374, 351.5534363, -193.1456299, 209.3964691, -556.5159302, 544.6990967
2: -504.3127136, 383.7380676, -278.1663513, 229.9753418, -734.2880859, 661.9044189
3: -195.8162994, 493.9827576, -117.0500183, 279.4069214, -475.2232056, 611.0327148
4: -562.0089111, 380.1941833, -311.8818665, 227.1875916, -789.1965332, 692.0759888

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3645095, upper bound: 804.9115190
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7074977, upper bound: 804.9121042
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -436.6898499, 365.1800842, -332.7971191, 281.7926941, -718.4825439, 697.9771729
1: -349.9454956, 354.0716248, -265.7110901, 274.8519287, -624.7973022, 619.7827148
2: -508.4296265, 386.5229187, -384.1240845, 300.9845886, -809.4141846, 770.6468506
3: -197.2324371, 497.9155884, -153.6954041, 379.7464600, -576.9788208, 651.6109619
4: -566.5614014, 382.8739319, -429.0441895, 297.0279236, -863.5893555, 811.9179688

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0975603, upper bound: 806.2167768
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1590872, upper bound: 806.2172850
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -376.6553345, 315.6994019, -238.6308594, 209.4359131, -586.0912476, 554.3302612
1: -301.6979370, 306.2880249, -189.5968323, 204.8708801, -506.5688171, 495.8848572
2: -437.8033752, 334.6196594, -272.8941040, 225.0126953, -662.8160400, 607.5137939
3: -170.2619171, 429.3719177, -114.6157455, 273.5653076, -443.8272095, 543.9876709
4: -487.2684631, 331.3125610, -305.7852173, 222.1712952, -709.4397583, 637.0977783

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8141806, upper bound: 805.5668972
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7252813, upper bound: 805.5118629
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.1428369, upper bound: 805.5116039
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -380.1496887, 318.2424622, -334.0037537, 281.0453186, -661.1948853, 652.2462158
1: -304.5241394, 308.7597961, -266.7456055, 274.1578979, -578.6820068, 575.5053711
2: -441.9195862, 337.3076477, -385.4673462, 300.1467285, -742.0662842, 722.7748413
3: -171.6442566, 433.2800598, -153.1459808, 380.6857300, -552.3299561, 586.4260254
4: -491.8168945, 333.9521179, -430.1876221, 296.0820312, -787.8989258, 764.1397705

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2360876, upper bound: 806.2167483
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1761113, upper bound: 806.2167634
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -433.1940002, 362.6026001, -243.7677917, 213.4292908, -646.6232910, 606.3703613
1: -347.1195374, 351.5534363, -193.7469482, 208.7287445, -555.8482666, 545.3003540
2: -504.3127136, 383.7380676, -278.9052429, 229.1844482, -733.4971924, 662.6431885
3: -195.8162994, 493.9827576, -116.8389282, 279.3439636, -475.1602783, 610.8216553
4: -562.0089111, 380.1941833, -312.4699707, 226.3032074, -788.3120117, 692.6639404

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8136292, upper bound: 805.5667001
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7728738, upper bound: 805.5563515
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -436.6898499, 365.1800842, -339.1815186, 285.2177429, -721.9075928, 704.3615723
1: -349.9454956, 354.0716248, -270.9281616, 278.1785583, -628.1240234, 624.9997559
2: -508.4296265, 386.5229187, -391.5458069, 304.4880066, -812.9176025, 778.0686646
3: -197.2324371, 497.9155884, -155.5240631, 386.5900574, -583.8222046, 653.4395752
4: -566.5614014, 382.8739319, -436.9592896, 300.3887939, -866.9501953, 819.8331909

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7838145, upper bound: 805.7832837
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3308730, upper bound: 805.7841391
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -394.8950500, 334.1777039, -394.8950500, 334.1777039, -729.0727539, 729.0727539
1: -316.1249084, 323.8029480, -316.1249084, 323.8029480, -639.9278564, 639.9278564
2: -459.4748840, 353.5050354, -459.4748840, 353.5050354, -812.9797974, 812.9797363
3: -179.7761841, 450.4103699, -179.7761841, 450.4103699, -630.1864624, 630.1864624
4: -512.8085938, 350.7399292, -512.8085938, 350.7399292, -863.5485229, 863.5485229

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1690196, upper bound: 808.4939443
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6970602, upper bound: 808.6956490
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -394.8950500, 334.1777039, -672.0262451, 545.8123169, -936.4828491, 1006.2039795
1: -316.1249084, 323.8029480, -540.2258301, 530.1713257, -842.3923950, 864.0286865
2: -459.4748840, 353.5050354, -781.6162720, 577.3313599, -1032.4698486, 1135.1213379
3: -179.7761841, 450.4103699, -297.4957581, 756.6786499, -936.4547119, 744.0138550
4: -512.8085938, 350.7399292, -869.2036743, 570.7238770, -1079.3920898, 1219.9433594

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1690196, upper bound: 808.4939371
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6970602, upper bound: 808.7492196
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -672.0262451, 545.8123169, -394.8950500, 334.1777039, -1006.2039795, 936.4827881
1: -540.2258301, 530.1713257, -316.1249084, 323.8029480, -864.0286865, 842.3923950
2: -781.6162720, 577.3313599, -459.4748840, 353.5050354, -1135.1213379, 1032.4699707
3: -297.4957581, 756.6786499, -179.7761841, 450.4103699, -744.0138550, 936.4547119
4: -869.2036743, 570.7238770, -512.8085938, 350.7399292, -1219.9433594, 1079.3920898

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9345310, upper bound: 808.8214705
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9885146, upper bound: 808.8212992
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -673.0687256, 546.6938477, -673.0687256, 546.6938477, -1210.4691162, 1210.4691162
1: -541.0684204, 531.0290527, -541.0684204, 531.0290527, -1064.6252441, 1064.6252441
2: -782.8427124, 578.2561035, -782.8427124, 578.2561035, -1352.5781250, 1352.5780029
3: -297.9616089, 757.8487549, -297.9616089, 757.8487549, -1050.4067383, 1050.4067383
4: -870.5672607, 571.6234131, -870.5672607, 571.6234131, -1432.8892822, 1432.8892822

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9345310, upper bound: 808.9889159
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9885146, upper bound: 808.9887758
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -394.8950500, 334.1777039, -400.8190918, 338.1311340, -733.0261841, 734.9968262
1: -316.1249084, 323.8029480, -321.0563354, 327.5602722, -643.6851807, 644.8591309
2: -459.4748840, 353.5050354, -466.4544678, 357.4834900, -816.9583740, 819.9594116
3: -179.7761841, 450.4103699, -182.1852417, 457.0790405, -636.8549805, 632.5954590
4: -512.8085938, 350.7399292, -520.2389526, 354.6555786, -867.4639893, 870.9788818

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1691019, upper bound: 808.2320062
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6971425, upper bound: 808.2325544
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -394.8950500, 334.1777039, -682.4237061, 553.3160400, -944.3145752, 1016.6014404
1: -316.1249084, 323.8029480, -548.6349487, 537.3023071, -849.7717285, 872.4377441
2: -459.4748840, 353.5050354, -793.6346436, 585.0888062, -1040.5039062, 1147.1396484
3: -179.7761841, 450.4103699, -301.6555481, 767.8587646, -947.6347656, 748.3998413
4: -512.8085938, 350.7399292, -882.3398438, 578.4310303, -1087.4119873, 1233.0797119

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1691019, upper bound: 808.3507022
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6971425, upper bound: 808.4941811
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -672.0267334, 545.8126831, -400.8190918, 338.1311340, -1010.1578369, 942.5018921
1: -540.2261963, 530.1716919, -321.0563354, 327.5602722, -867.7863770, 847.3952637
2: -781.6167603, 577.3316650, -466.4544678, 357.4834900, -1139.1002197, 1039.5976562
3: -297.4959717, 756.6791992, -182.1852417, 457.0790405, -750.7524414, 938.8643799
4: -869.2042847, 570.7242432, -520.2389526, 354.6555786, -1223.8598633, 1086.9887695

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9345601, upper bound: 808.2536627
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9885437, upper bound: 808.2534936
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -673.0687256, 546.6938477, -683.3704834, 554.1118774, -1218.2207031, 1220.2379150
1: -541.0684204, 531.0290527, -549.4003906, 538.0770874, -1071.9255371, 1072.5794678
2: -782.8427124, 578.2561035, -794.7464600, 585.9254150, -1360.5286865, 1363.9803467
3: -297.9616089, 757.8487549, -302.0794983, 768.9109497, -1061.3103027, 1054.7530518
4: -870.5672607, 571.6234131, -883.5769653, 579.2427979, -1440.8267822, 1445.3806152

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9345601, upper bound: 808.3419851
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9885437, upper bound: 808.3418138
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -400.8190918, 338.1311340, -394.8950500, 334.1777039, -734.9968262, 733.0261841
1: -321.0563354, 327.5602722, -316.1249084, 323.8029480, -644.8590698, 643.6851807
2: -466.4544678, 357.4834900, -459.4748840, 353.5050354, -819.9594116, 816.9583740
3: -182.1852417, 457.0790405, -179.7761841, 450.4103699, -632.5955200, 636.8549805
4: -520.2389526, 354.6555786, -512.8085938, 350.7399292, -870.9788818, 867.4640503

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36

Time for candidate selection: 4.04 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6064102, upper bound: 808.2048217
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1889446, upper bound: 808.2054366
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -400.8190918, 338.1311340, -672.0267334, 545.8126831, -942.5018921, 1010.1578369
1: -321.0563354, 327.5602722, -540.2261963, 530.1716919, -847.3952637, 867.7864380
2: -466.4544678, 357.4834900, -781.6167603, 577.3316650, -1039.5975342, 1139.1002197
3: -182.1852417, 457.0790405, -297.4959717, 756.6791992, -938.8643188, 750.7523804
4: -520.2389526, 354.6555786, -869.2042847, 570.7242432, -1086.9886475, 1223.8598633

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36

Time for candidate selection: 4.15 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6064305, upper bound: 808.2048302
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1889446, upper bound: 808.2054281
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -682.4237061, 553.3160400, -394.8950500, 334.1777039, -1016.6014404, 944.3145142
1: -548.6349487, 537.3023071, -316.1249084, 323.8029480, -872.4378662, 849.7717896
2: -793.6346436, 585.0888062, -459.4748840, 353.5050354, -1147.1396484, 1040.5040283
3: -301.6555481, 767.8587646, -179.7761841, 450.4103699, -748.3999023, 947.6347656
4: -882.3398438, 578.4310303, -512.8085938, 350.7399292, -1233.0797119, 1087.4119873

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5461097, upper bound: 808.8218318
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415442, upper bound: 808.5260438
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -683.3941650, 554.1363525, -673.0687256, 546.6938477, -1220.2613525, 1218.2432861
1: -549.4193726, 538.1006470, -541.0684204, 531.0290527, -1072.5985107, 1071.9476318
2: -794.7759399, 585.9496460, -782.8427124, 578.2561035, -1364.0098877, 1360.5517578
3: -302.0893250, 768.9462280, -297.9616089, 757.8487549, -1054.7628174, 1061.3447266
4: -883.6091919, 579.2677612, -870.5672607, 571.6234131, -1445.4130859, 1440.8502197

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5461097, upper bound: 808.9891401
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415442, upper bound: 808.5260538
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -400.8190918, 338.1311340, -400.8190918, 338.1311340, -738.9501953, 738.9501953
1: -321.0563354, 327.5602722, -321.0563354, 327.5602722, -648.6164551, 648.6164551
2: -466.4544678, 357.4834900, -466.4544678, 357.4834900, -823.9379883, 823.9379883
3: -182.1852417, 457.0790405, -182.1852417, 457.0790405, -639.2640381, 639.2640381
4: -520.2389526, 354.6555786, -520.2389526, 354.6555786, -874.8943481, 874.8944092

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0667372, upper bound: 808.0956454
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532170, upper bound: 808.2531693
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -400.8190918, 338.1311340, -682.4240723, 553.3163452, -950.3336792, 1020.5551758
1: -321.0563354, 327.5602722, -548.6352539, 537.3026123, -854.7744751, 876.1954346
2: -466.4544678, 357.4834900, -793.6350708, 585.0892944, -1047.6317139, 1151.1184082
3: -182.1852417, 457.0790405, -301.6557922, 767.8592529, -950.0444336, 755.1383057
4: -520.2389526, 354.6555786, -882.3404541, 578.4313965, -1095.0086670, 1236.9959717

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0667372, upper bound: 808.0956562
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532170, upper bound: 808.2640626
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -682.4240723, 553.3163452, -400.8190918, 338.1311340, -1020.5551758, 950.3336182
1: -548.6352539, 537.3026123, -321.0563354, 327.5602722, -876.1954956, 854.7744751
2: -793.6350708, 585.0892944, -466.4544678, 357.4834900, -1151.1184082, 1047.6317139
3: -301.6557922, 767.8592529, -182.1852417, 457.0790405, -755.1382446, 950.0443726
4: -882.3404541, 578.4313965, -520.2389526, 354.6555786, -1236.9959717, 1095.0086670

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5461029, upper bound: 808.2540262
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415373, upper bound: 808.2533698
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -683.3941650, 554.1363525, -683.3704834, 554.1118774, -1228.0128174, 1228.0118408
1: -549.4193726, 538.1006470, -549.4003906, 538.0770874, -1079.8988037, 1079.9017334
2: -794.7759399, 585.9496460, -794.7464600, 585.9254150, -1371.9604492, 1371.9539795
3: -302.0893250, 768.9462280, -302.0794983, 768.9109497, -1065.6663818, 1065.6910400
4: -883.6091919, 579.2677612, -883.5769653, 579.2427979, -1453.3504639, 1453.3414307

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5461027, upper bound: 808.3423465
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415373, upper bound: 808.3416901
time: 0.72 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.78 seconds
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.2360876, upper bound: 806.2167483
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.1762979, upper bound: 806.2167634
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 4, lower bound: -807.3645095, upper bound: 804.9115190
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -807.7074977, upper bound: 804.9121042
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.0975603, upper bound: 806.2167768
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.1590872, upper bound: 806.2172850
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 4, lower bound: -806.7252813, upper bound: 805.5118629
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 4, lower bound: -806.1428369, upper bound: 805.5116039
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.2360876, upper bound: 806.2167483
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.1761113, upper bound: 806.2167634
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -807.8136292, upper bound: 805.5667001
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -807.7728738, upper bound: 805.5563515
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 4, lower bound: -805.7838145, upper bound: 805.7832837
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 4, lower bound: -807.3308730, upper bound: 805.7841391
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.1690196, upper bound: 808.4939443
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.6970602, upper bound: 808.6956490
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.1690196, upper bound: 808.4939371
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.6970602, upper bound: 808.7492196
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.9345310, upper bound: 808.8214705
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.9885146, upper bound: 808.8212992
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.9345310, upper bound: 808.9889159
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.9885146, upper bound: 808.9887758
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.1691019, upper bound: 808.2320062
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.6971425, upper bound: 808.2325544
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.1691019, upper bound: 808.3507022
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.6971425, upper bound: 808.4941811
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.9345601, upper bound: 808.2536627
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.9885437, upper bound: 808.2534936
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.9345601, upper bound: 808.3419851
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.9885437, upper bound: 808.3418138
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -807.6064102, upper bound: 808.2048217
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.1889446, upper bound: 808.2054366
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -807.6064305, upper bound: 808.2048302
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.1889446, upper bound: 808.2054281
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.5461097, upper bound: 808.8218318
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.3415442, upper bound: 808.5260438
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.5461097, upper bound: 808.9891401
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.3415442, upper bound: 808.5260538
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.0667372, upper bound: 808.0956454
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.2532170, upper bound: 808.2531693
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.0667372, upper bound: 808.0956562
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.2532170, upper bound: 808.2640626
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.5461029, upper bound: 808.2540262
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.3415373, upper bound: 808.2533698
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.5461027, upper bound: 808.3423465
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 4, lower bound: -808.3415373, upper bound: 808.3416901

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -265.4413147, 234.5291290, -326.8363037, 277.0902710, -542.5316162, 561.3654175
1: -212.0245361, 227.3792267, -260.8926392, 270.3126831, -482.3371582, 488.2717896
2: -308.5409546, 248.7822266, -377.0998230, 296.0925293, -604.6333618, 625.8819580
3: -124.6852646, 307.7318726, -151.0160370, 372.9862366, -497.6714172, 458.7479248
4: -343.8453064, 246.8465729, -421.2131653, 292.1578064, -636.0031128, 668.0596313

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2354751, upper bound: 806.2166840
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8318200, upper bound: 806.0902486
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8545873, upper bound: 804.4561279
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2360662, upper bound: 806.2167483
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -368.1410522, 308.8850098, -327.1813660, 277.3379822, -645.4790039, 636.0662231
1: -294.9066162, 299.8633118, -261.1712952, 270.5529480, -565.4595337, 561.0346069
2: -427.6830444, 327.7184753, -377.5047913, 296.3498230, -724.0327148, 705.2232666
3: -166.5783997, 420.1256409, -151.1579590, 373.3670654, -539.9453735, 571.2835693
4: -475.9859619, 324.2613831, -421.6633911, 292.4135742, -768.3994751, 745.9245605

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1758236, upper bound: 806.2166031
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1758942, upper bound: 805.9763455
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 26

Time for candidate selection: 5.53 seconds

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1762979, upper bound: 806.2167634
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1253373, upper bound: 805.6219604
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1735392, upper bound: 804.3024402
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1736615, upper bound: 806.2163487
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -421.8020935, 353.0725708, -243.2164001, 214.0600586, -635.8621216, 596.2887573
1: -337.9725342, 342.4955139, -193.1456299, 209.3964691, -547.3689575, 535.6411133
2: -490.7892761, 374.1408997, -278.1663513, 229.9753418, -720.7645874, 652.3071899
3: -190.8523712, 481.2301636, -117.0500183, 279.4069214, -470.2592773, 598.2801514
4: -546.9939575, 370.5241394, -311.8818665, 227.1875916, -774.1815186, 682.4059448

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6628989, upper bound: 804.6417962
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 7

Time for candidate selection: 4.53 seconds

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6957652, upper bound: 804.9121042
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5087058, upper bound: 804.1235708
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6821899, upper bound: 802.9189597
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6787366, upper bound: 804.9118346
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -301.9802551, 266.9254761, -332.4513855, 281.5437012, -583.5238647, 599.3767090
1: -241.3724365, 258.4585876, -265.4318848, 274.6106262, -515.9830322, 523.8903809
2: -351.4915771, 282.9038696, -383.7181396, 300.7259827, -652.2174683, 666.6218262
3: -142.9042206, 349.7152405, -153.5532990, 379.3640442, -522.2681885, 503.2685547
4: -392.2167969, 280.8281860, -428.5929260, 296.7707520, -688.9874268, 709.4210815

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0971487, upper bound: 806.2165725
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1664093, upper bound: 805.0724477
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0975167, upper bound: 806.2167768
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -425.2669678, 355.6134644, -332.7971191, 281.7926941, -707.0596924, 688.4105835
1: -340.7726135, 345.0171204, -265.7110901, 274.8519287, -615.6243286, 610.7282104
2: -494.8684692, 376.8844604, -384.1240845, 300.9845886, -795.8530273, 761.0085449
3: -192.2481842, 485.1243896, -153.6954041, 379.7464600, -571.9946289, 638.8198242
4: -551.5020752, 373.2102356, -429.0441895, 297.0279236, -848.5300293, 802.2543945

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1586576, upper bound: 806.2166004
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5734485, upper bound: 805.0730753
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1590420, upper bound: 806.2172850
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -265.4413147, 234.5291290, -333.6734314, 280.8059387, -546.2472534, 568.2025757
1: -212.0245361, 227.3792267, -266.4792175, 273.9260864, -485.9505920, 493.8583984
2: -308.5409546, 248.7822266, -385.0820312, 299.8981018, -608.4389038, 633.8641357
3: -124.6852646, 307.7318726, -153.0087585, 380.3206177, -505.0058594, 460.7406311
4: -343.8453064, 246.8465729, -429.7599182, 295.8359070, -639.6812134, 676.6064453

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2354751, upper bound: 806.2166843
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8317119, upper bound: 806.0907377
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2354608, upper bound: 805.9763304
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5916141, upper bound: 805.7832939
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -368.1410522, 308.8850098, -334.0037537, 281.0453186, -649.1864014, 642.8887329
1: -294.9066162, 299.8633118, -266.7456055, 274.1578979, -569.0645142, 566.6088867
2: -427.6830444, 327.7184753, -385.4673462, 300.1467285, -727.8296509, 713.1857910
3: -166.5783997, 420.1256409, -153.1459808, 380.6857300, -547.2641602, 573.2716064
4: -475.9859619, 324.2613831, -430.1876221, 296.0820312, -772.0679932, 754.4489136

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1756856, upper bound: 806.2166031
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1756628, upper bound: 805.9763455
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35

Time for candidate selection: 5.47 seconds

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1252805, upper bound: 805.5879956
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1761113, upper bound: 806.2167634
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 10

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1735356, upper bound: 804.3438266
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1736073, upper bound: 806.2163487
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -298.5444946, 264.5039673, -243.4473877, 213.2043152, -511.7487793, 507.9513550
1: -238.5886078, 256.1078186, -193.4880829, 208.5100708, -447.0986938, 449.5958862
2: -347.4354248, 280.3396912, -278.5323792, 228.9493256, -576.3847656, 558.8720703
3: -141.5732880, 345.9049683, -116.7136536, 278.9945374, -420.5678101, 462.6186218
4: -387.7435303, 278.3094482, -312.0572815, 226.0706024, -613.8141479, 590.3666992

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5369767, upper bound: 805.2611863
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8132110, upper bound: 805.2824878
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35

Time for candidate selection: 5.32 seconds

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8136292, upper bound: 805.5565418
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5576963, upper bound: 805.5667001
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -421.8020935, 353.0725708, -243.7677917, 213.4292908, -635.2313843, 596.8401489
1: -337.9725342, 342.4955139, -193.7469482, 208.7287445, -546.7012329, 536.2423706
2: -490.7892761, 374.1408997, -278.9052429, 229.1844482, -719.9737549, 653.0460205
3: -190.8523712, 481.2301636, -116.8389282, 279.3439636, -470.1963501, 598.0690918
4: -546.9939575, 370.5241394, -312.4699707, 226.3032074, -773.2971191, 682.9939575

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7669761, upper bound: 805.2222133
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35

Time for candidate selection: 5.03 seconds

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7728738, upper bound: 805.5563455
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.0834637, upper bound: 805.5563515
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -366.3092041, 313.3845520, -394.8950500, 334.1777039, -700.4869385, 708.2796021
1: -293.1076965, 303.4456177, -316.1249084, 323.8029480, -616.9104004, 619.5705566
2: -426.1185608, 331.2268677, -459.4748840, 353.5050354, -779.6234131, 790.7017822
3: -168.1203613, 419.2138367, -179.7761841, 450.4103699, -618.5306396, 598.9897461
4: -475.9714966, 329.0264893, -512.8085938, 350.7399292, -826.7114258, 841.8350830

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1686534, upper bound: 808.1682240
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1686534, upper bound: 808.4946725
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -464.3193970, 387.3415527, -394.8950500, 334.1777039, -798.4970703, 782.2365723
1: -372.3395691, 375.2162781, -316.1249084, 323.8029480, -696.1423950, 691.3411865
2: -540.9116211, 409.2961121, -459.4748840, 353.5050354, -894.4165039, 868.7709961
3: -210.2126617, 528.4973145, -179.7761841, 450.4103699, -660.6230469, 708.2734375
4: -603.1533203, 406.5039062, -512.8085938, 350.7399292, -953.8932495, 919.3124390

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6966942, upper bound: 808.1689398
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6966942, upper bound: 808.6956470
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -366.3092041, 313.3845520, -671.6701660, 545.5111694, -907.4331055, 985.0546875
1: -293.1076965, 303.4456177, -539.9378052, 529.8783569, -818.9158325, 843.3834229
2: -426.1185608, 331.2268677, -781.1973267, 577.0155029, -998.4694824, 1112.4241943
3: -168.1203613, 419.2138367, -297.3366699, 756.2789307, -924.3992310, 712.3128052
4: -475.9714966, 329.0264893, -868.7379150, 570.4166260, -1041.9016113, 1197.7644043

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1692495, upper bound: 808.3671989
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1692429, upper bound: 808.4939443
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -464.3193970, 387.3415527, -671.6701660, 545.5111694, -1006.0194092, 1059.0114746
1: -372.3395691, 375.2162781, -539.9378052, 529.8783569, -898.4847412, 915.1540527
2: -540.9116211, 409.2961121, -781.1973267, 577.0155029, -1113.7731934, 1190.4934082
3: -210.2126617, 528.4973145, -297.3366699, 756.2789307, -966.4915771, 821.9914551
4: -603.1533203, 406.5039062, -868.7379150, 570.4166260, -1169.7347412, 1275.2416992

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6975592, upper bound: 808.6374778
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6974082, upper bound: 808.7492118
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -584.9732666, 474.7602234, -388.6331482, 329.0883789, -914.0616455, 858.1783447
1: -469.9567871, 461.3920898, -311.0547791, 318.8909912, -788.8476562, 767.7312622
2: -679.3198242, 502.9391785, -452.0673828, 348.1708984, -1027.4906006, 949.6303711
3: -259.6007996, 656.3446045, -176.9993439, 443.2226868, -698.1658325, 833.3439331
4: -755.2292480, 497.0121155, -504.5723877, 345.4270325, -1100.6562500, 996.3770752

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9340526, upper bound: 808.1695727
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9344188, upper bound: 808.6976133
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -665.7241211, 540.1499634, -394.8950500, 334.1777039, -999.9018555, 931.2462769
1: -535.1021118, 524.7368774, -316.1249084, 323.8029480, -858.9049072, 837.2993774
2: -774.1378174, 571.4366455, -459.4748840, 353.5050354, -1127.6428223, 1027.0078125
3: -294.4309692, 749.3394775, -179.7761841, 450.4103699, -741.2810669, 929.1155396
4: -860.9558105, 564.7789917, -512.8085938, 350.7399292, -1211.6958008, 1073.9160156

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9754430, upper bound: 808.1694014
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9758092, upper bound: 808.6974420
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -586.0347900, 475.6571045, -666.2177124, 541.1524658, -1117.4766846, 1131.6470947
1: -470.8153687, 462.2648315, -535.5295410, 525.6744995, -988.6051025, 989.5496826
2: -680.5691528, 503.8807678, -774.7819214, 572.4594727, -1244.1506348, 1269.1654053
3: -260.0764771, 657.5344238, -294.9795532, 749.9725342, -1003.8344727, 947.0281372
4: -756.6181641, 497.9292908, -861.5971680, 565.8871460, -1313.0258789, 1349.1868896

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9348007, upper bound: 808.9345820
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9348007, upper bound: 808.9887758
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -666.8051758, 541.0629883, -673.0687256, 546.6938477, -1204.2736816, 1205.2607422
1: -535.9760132, 525.6253662, -541.0684204, 531.0290527, -1059.5598145, 1059.5598145
2: -775.4098511, 572.3945923, -782.8427124, 578.2561035, -1345.2147217, 1347.1456299
3: -294.9136658, 750.5516357, -297.9616089, 757.8487549, -1047.6884766, 1043.1730957
4: -862.3701172, 565.7105103, -870.5672607, 571.6234131, -1424.8131104, 1427.4416504

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9887625, upper bound: 808.9345821
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9887625, upper bound: 808.9887758
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -366.3092041, 313.3845520, -400.8190918, 338.1311340, -704.4403076, 714.2036133
1: -293.1076965, 303.4456177, -321.0563354, 327.5602722, -620.6678467, 624.5019531
2: -426.1185608, 331.2268677, -466.4544678, 357.4834900, -783.6019897, 797.6813354
3: -168.1203613, 419.2138367, -182.1852417, 457.0790405, -625.1991577, 601.3988037
4: -475.9714966, 329.0264893, -520.2389526, 354.6555786, -830.6269531, 849.2654419

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1690706, upper bound: 808.0493823
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1690748, upper bound: 808.2320062
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -464.3193970, 387.3415527, -400.8190918, 338.1311340, -802.4505005, 788.1605835
1: -372.3395691, 375.2162781, -321.0563354, 327.5602722, -699.8997803, 696.2725830
2: -540.9116211, 409.2961121, -466.4544678, 357.4834900, -898.3950806, 875.7506104
3: -210.2126617, 528.4973145, -182.1852417, 457.0790405, -667.2916260, 710.6824951
4: -603.1533203, 406.5039062, -520.2389526, 354.6555786, -957.8088379, 926.7427368

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6971112, upper bound: 808.0498731
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6971154, upper bound: 808.2325544
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -366.3092041, 313.3845520, -682.0822754, 553.0276489, -915.2772217, 995.4667969
1: -293.1076965, 303.4456177, -548.3588257, 537.0216064, -826.3070679, 851.8044434
2: -426.1185608, 331.2268677, -793.2331543, 584.7861938, -1006.5162354, 1124.4599609
3: -168.1203613, 419.2138367, -301.5029297, 767.4763184, -935.5965576, 716.7050171
4: -475.9714966, 329.0264893, -881.8936157, 578.1365967, -1049.9342041, 1210.9200439

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1697269, upper bound: 808.3507056
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1692352, upper bound: 808.3313348
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -464.3193970, 387.3415527, -682.0822754, 553.0276489, -1013.8636475, 1069.4237061
1: -372.3395691, 375.2162781, -548.3588257, 537.0216064, -905.8759766, 923.5750732
2: -540.9116211, 409.2961121, -793.2331543, 584.7861938, -1121.8203125, 1202.5291748
3: -210.2126617, 528.4973145, -301.5029297, 767.4763184, -977.6889648, 826.3837891
4: -603.1533203, 406.5039062, -881.8936157, 578.1365967, -1177.7673340, 1288.3973389

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6979456, upper bound: 808.4941850
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6972758, upper bound: 808.3319447
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -584.9736938, 474.7606201, -394.6436768, 333.0936584, -918.0673828, 864.2933350
1: -469.9571533, 461.3924866, -316.0532227, 322.6996155, -792.6567383, 772.8078003
2: -679.3203125, 502.9395447, -459.1423340, 352.2072144, -1031.5273438, 956.8625488
3: -259.6009521, 656.3451538, -179.4498444, 449.9688721, -704.9865723, 835.7949829
4: -755.2297363, 497.0124512, -512.1105957, 349.3966980, -1104.6264648, 1004.0924072

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9346273, upper bound: 808.0669831
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9346273, upper bound: 808.2534936
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -665.7244873, 540.1502686, -400.8190918, 338.1311340, -1003.8555908, 937.2654419
1: -535.1024780, 524.7371826, -321.0563354, 327.5602722, -862.6625366, 842.3021851
2: -774.1382446, 571.4370728, -466.4544678, 357.4834900, -1131.6217041, 1034.1356201
3: -294.4311523, 749.3399658, -182.1852417, 457.0790405, -748.0194702, 931.5251465
4: -860.9564209, 564.7793579, -520.2389526, 354.6555786, -1215.6120605, 1081.5126953

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9886109, upper bound: 808.0669831
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9886109, upper bound: 808.2534936
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -586.0347900, 475.6571045, -676.5567627, 548.6171875, -1125.2667236, 1141.4517822
1: -470.8153687, 462.2648315, -543.8908691, 532.7670898, -995.9439087, 997.5325928
2: -680.5691528, 503.8807678, -786.7361450, 580.1801758, -1252.1427002, 1280.6107178
3: -260.0764771, 657.5344238, -299.1295776, 761.0806885, -1014.7806396, 951.4012451
4: -756.6181641, 497.9292908, -874.6638794, 573.5538940, -1320.9992676, 1361.6915283

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9346769, upper bound: 808.3418138
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9346769, upper bound: 808.3418138
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -666.8051758, 541.0629883, -683.3704834, 554.1118774, -1212.0252686, 1215.0294189
1: -535.9760132, 525.6253662, -549.4003906, 538.0770874, -1066.8602295, 1067.5141602
2: -775.4098511, 572.3945923, -794.7464600, 585.9254150, -1353.1654053, 1358.5478516
3: -294.9136658, 750.5516357, -302.0794983, 768.9109497, -1058.5920410, 1047.5195312
4: -862.3701172, 565.7105103, -883.5769653, 579.2427979, -1432.7507324, 1439.9329834

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9886604, upper bound: 808.3418138
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9886604, upper bound: 808.3418138
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -374.9313354, 319.5372620, -394.8950500, 334.1777039, -709.1090088, 714.4323120
1: -299.9557800, 309.3746338, -316.1249084, 323.8029480, -623.7587280, 625.4995117
2: -435.4627380, 337.8553162, -459.4748840, 353.5050354, -788.9676514, 797.3301392
3: -171.5083618, 427.6545715, -179.7761841, 450.4103699, -621.9187012, 607.4306641
4: -486.4178772, 335.3886108, -512.8085938, 350.7399292, -837.1578369, 848.1971436

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6084465, upper bound: 807.7966025
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6092288, upper bound: 808.0857685
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -426.3991089, 375.7694397, -394.8950500, 334.1777039, -760.5767822, 770.6644897
1: -341.9978027, 364.2934570, -316.1249084, 323.8029480, -665.8006592, 680.4183350
2: -496.7396851, 397.4942932, -459.4748840, 353.5050354, -850.2446899, 856.9691162
3: -201.3342896, 490.2865906, -179.7761841, 450.4103699, -651.7446289, 670.0627441
4: -553.4467773, 392.3744507, -512.8085938, 350.7399292, -904.1867065, 905.1829834

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0738291, upper bound: 807.7973604
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0746360, upper bound: 808.0865263
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -374.9313354, 319.5372620, -672.0267334, 545.8126831, -916.4457397, 991.5639648
1: -299.9557800, 309.3746338, -540.2261963, 530.1716919, -826.1491089, 849.6008301
2: -435.4627380, 337.8553162, -781.6167603, 577.3316650, -1008.3027344, 1119.4719238
3: -171.5083618, 427.6545715, -297.4959717, 756.6791992, -928.1875610, 721.1955566
4: -486.4178772, 335.3886108, -869.2042847, 570.7242432, -1052.8447266, 1204.5928955

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 11

Time for candidate selection: 4.20 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6052850, upper bound: 807.6219159
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6052850, upper bound: 808.2048302
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -426.3991089, 375.7694397, -672.0267334, 545.8126831, -967.3699341, 1047.7961426
1: -341.9978027, 364.2934570, -540.2261963, 530.1716919, -867.7597046, 904.5196533
2: -496.7396851, 397.4942932, -781.6167603, 577.3316650, -1069.0056152, 1179.1110840
3: -201.3342896, 490.2865906, -297.4959717, 756.6791992, -958.0134888, 784.0430298
4: -553.4467773, 392.3744507, -869.2042847, 570.7242432, -1119.1583252, 1261.5787354

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 11

Time for candidate selection: 4.25 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1885005, upper bound: 807.6225223
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1885005, upper bound: 808.2054281
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -598.1709595, 484.5003662, -388.6331482, 329.0883789, -927.2593384, 868.2177124
1: -480.6658325, 470.6712036, -311.0547791, 318.8909912, -799.5566406, 777.2412720
2: -694.7885132, 513.0001221, -452.0673828, 348.1708984, -1042.9594727, 959.9501343
3: -264.9202271, 670.8148193, -176.9993439, 443.2226868, -703.7210083, 847.8141479
4: -772.1105957, 507.0472717, -504.5723877, 345.4270325, -1117.5375977, 1006.6704102

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5456313, upper bound: 808.1699177
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5459975, upper bound: 808.6979583
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -675.9480591, 547.5113525, -394.8950500, 334.1777039, -1010.1257324, 938.9359741
1: -543.3765869, 531.7075806, -316.1249084, 323.8029480, -867.1793823, 844.5195923
2: -785.9476929, 579.0417480, -459.4748840, 353.5050354, -1139.4527588, 1034.8937988
3: -298.4931641, 760.3677979, -179.7761841, 450.4103699, -745.5774536, 940.1437988
4: -873.8426514, 572.3395386, -512.8085938, 350.7399292, -1224.5822754, 1081.7939453

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3316407, upper bound: 808.1664969
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3320069, upper bound: 808.4901890
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -599.1633301, 485.3391724, -666.2177124, 541.1524658, -1130.1210938, 1141.6326904
1: -481.4682617, 471.4866333, -535.5295410, 525.6744995, -998.9084473, 999.0057373
2: -695.9554443, 513.8804932, -774.7819214, 572.4594727, -1258.9827881, 1279.4277344
3: -265.3652039, 671.9259033, -294.9795532, 749.9725342, -1009.3634644, 961.2507935
4: -773.4079590, 507.9051819, -861.5971680, 565.8871460, -1329.2091064, 1359.4318848

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3407760, upper bound: 808.3695855
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3407760, upper bound: 808.5260438
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -676.9556274, 548.3621216, -673.0687256, 546.6938477, -1213.9268799, 1212.8919678
1: -544.1911011, 532.5355835, -541.0684204, 531.0290527, -1067.4180908, 1066.7226562
2: -787.1326904, 579.9343872, -782.8427124, 578.2561035, -1356.4470215, 1354.9699707
3: -298.9427490, 761.4962158, -297.9616089, 757.8487549, -1051.9543457, 1053.9605713
4: -875.1602783, 573.2073975, -870.5672607, 571.6234131, -1437.0645752, 1435.2593994

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3407760, upper bound: 808.3695855
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3407760, upper bound: 808.5260438
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -348.8487854, 294.2696228, -394.6436768, 333.0936584, -681.9423828, 688.9132690
1: -279.3065796, 285.4207153, -316.0532227, 322.6996155, -602.0061035, 601.4739380
2: -405.0474854, 311.6168823, -459.1423340, 352.2072144, -757.2546387, 770.7592163
3: -158.3417816, 397.9795532, -179.4498444, 449.9688721, -608.3106079, 577.4293823
4: -451.2043457, 308.8600769, -512.1105957, 349.3966980, -800.6008911, 820.9706421

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0657336, upper bound: 808.0657503
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0657335, upper bound: 808.1705309
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -392.7154541, 331.1437073, -400.8190918, 338.1311340, -730.8465576, 731.9627686
1: -314.4709778, 320.8226929, -321.0563354, 327.5602722, -642.0311890, 641.8788452
2: -456.8787231, 350.2012329, -466.4544678, 357.4834900, -814.3621826, 816.6557007
3: -178.5253296, 447.7351379, -182.1852417, 457.0790405, -635.6042480, 629.9202881
4: -509.6759033, 347.4036255, -520.2389526, 354.6555786, -864.3314209, 867.6425781

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0660411, upper bound: 808.0666594
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0660348, upper bound: 808.2531693
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -348.8487854, 294.2696228, -675.5661621, 547.7834473, -891.8169556, 969.8357544
1: -279.3065796, 285.4207153, -543.0901489, 531.9554443, -806.8845825, 828.5108643
2: -405.0474854, 311.6168823, -785.5729980, 579.3040771, -979.4238281, 1097.1895752
3: -158.3417816, 397.9795532, -298.6861877, 759.9782715, -918.3197632, 692.4665527
4: -451.2043457, 308.8600769, -873.3699341, 572.7032471, -1019.1015625, 1182.2299805

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0662031, upper bound: 808.0803281
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0662028, upper bound: 808.0956454
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -392.7154541, 331.1437073, -682.3615723, 553.2636719, -941.6582031, 1013.5052490
1: -314.4709778, 320.8226929, -548.5847168, 537.2512817, -847.7120972, 869.4072876
2: -456.8787231, 350.2012329, -793.5615234, 585.0338135, -1037.3306885, 1143.7626953
3: -178.5253296, 447.7351379, -301.6278076, 767.7893066, -946.3145752, 745.4942627
4: -509.6759033, 347.4036255, -882.2587280, 578.3774414, -1083.7052002, 1229.6623535

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0762917, upper bound: 808.0890156
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0762943, upper bound: 808.2640592
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -598.1713867, 484.5007629, -394.6436768, 333.0936584, -931.2650146, 874.3327026
1: -480.6661987, 470.6716003, -316.0532227, 322.6996155, -803.3656006, 782.3179932
2: -694.7891235, 513.0004883, -459.1423340, 352.2072144, -1046.9963379, 967.1821899
3: -264.9204102, 670.8153076, -179.4498444, 449.9688721, -710.5418701, 850.2651367
4: -772.1110229, 507.0476379, -512.1105957, 349.3966980, -1121.5076904, 1014.3858032

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0664789, upper bound: 808.0667754
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0664789, upper bound: 808.2533698
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -675.9484863, 547.5117188, -400.8190918, 338.1311340, -1014.0795898, 944.9550781
1: -543.3769531, 531.7079468, -321.0563354, 327.5602722, -870.9371338, 849.5223389
2: -785.9483032, 579.0421753, -466.4544678, 357.4834900, -1143.4317627, 1042.0214844
3: -298.4933777, 760.3682251, -182.1852417, 457.0790405, -752.3158569, 942.5532837
4: -873.8431396, 572.3399658, -520.2389526, 354.6555786, -1228.4987793, 1089.3905029

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0664789, upper bound: 808.0667754
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0664829, upper bound: 808.2533698
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -599.1633301, 485.3391724, -676.5567627, 548.6171875, -1137.9112549, 1151.4373779
1: -481.4682617, 471.4866333, -543.8908691, 532.7670898, -1006.2472534, 1006.9886475
2: -695.9554443, 513.8804932, -786.7361450, 580.1801758, -1266.9749756, 1290.8728027
3: -265.3652039, 671.9259033, -299.1295776, 761.0806885, -1020.3095703, 965.6239014
4: -773.4079590, 507.9051819, -874.6638794, 573.5538940, -1337.1823730, 1371.9365234

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3125838, upper bound: 808.2838647
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3287463, upper bound: 808.3258292
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -676.9556274, 548.3621216, -683.3704834, 554.1118774, -1221.6783447, 1222.6606445
1: -544.1911011, 532.5355835, -549.4003906, 538.0770874, -1074.7183838, 1074.6768799
2: -787.1326904, 579.9343872, -794.7464600, 585.9254150, -1364.3977051, 1366.3720703
3: -298.9427490, 761.4962158, -302.0794983, 768.9109497, -1062.8579102, 1058.3067627
4: -875.1602783, 573.2073975, -883.5769653, 579.2427979, -1445.0019531, 1447.7507324

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3407760, upper bound: 808.3416901
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3407760, upper bound: 808.3416901
time: 0.64 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.69 seconds
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.69
Output dim: 4, lower bound: -805.8545873, upper bound: 804.4561279
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.2360662, upper bound: 806.2167483
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1735392, upper bound: 804.3024402
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1736615, upper bound: 806.2163487
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.6821899, upper bound: 802.9189597
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.6787366, upper bound: 804.9118346
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.1664093, upper bound: 805.0724477
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0975167, upper bound: 806.2167768
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.5734485, upper bound: 805.0730753
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1590420, upper bound: 806.2172850
IS_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.2354608, upper bound: 805.9763304
IS_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.5916141, upper bound: 805.7832939
IS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1735356, upper bound: 804.3438266
IS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1736073, upper bound: 806.2163487
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.8136292, upper bound: 805.5565418
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.5576963, upper bound: 805.5667001
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.7728738, upper bound: 805.5563455
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.0834637, upper bound: 805.5563515
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1686534, upper bound: 808.1682240
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1686534, upper bound: 808.4946725
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.6966942, upper bound: 808.1689398
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.6966942, upper bound: 808.6956470
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1692495, upper bound: 808.3671989
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1692429, upper bound: 808.4939443
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.6975592, upper bound: 808.6374778
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.6974082, upper bound: 808.7492118
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9340526, upper bound: 808.1695727
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9344188, upper bound: 808.6976133
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9754430, upper bound: 808.1694014
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9758092, upper bound: 808.6974420
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9348007, upper bound: 808.9345820
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9348007, upper bound: 808.9887758
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9887625, upper bound: 808.9345821
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9887625, upper bound: 808.9887758
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1690706, upper bound: 808.0493823
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1690748, upper bound: 808.2320062
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.6971112, upper bound: 808.0498731
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.6971154, upper bound: 808.2325544
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1697269, upper bound: 808.3507056
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1692352, upper bound: 808.3313348
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.6979456, upper bound: 808.4941850
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.6972758, upper bound: 808.3319447
IS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9346273, upper bound: 808.0669831
IS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9346273, upper bound: 808.2534936
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9886109, upper bound: 808.0669831
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9886109, upper bound: 808.2534936
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9346769, upper bound: 808.3418138
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9346769, upper bound: 808.3418138
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9886604, upper bound: 808.3418138
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.9886604, upper bound: 808.3418138
IS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.6084465, upper bound: 807.7966025
IS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.6092288, upper bound: 808.0857685
IS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0738291, upper bound: 807.7973604
IS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0746360, upper bound: 808.0865263
IS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.6052850, upper bound: 807.6219159
IS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -807.6052850, upper bound: 808.2048302
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1885005, upper bound: 807.6225223
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.1885005, upper bound: 808.2054281
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.5456313, upper bound: 808.1699177
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.5459975, upper bound: 808.6979583
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.3316407, upper bound: 808.1664969
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.3320069, upper bound: 808.4901890
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.3407760, upper bound: 808.3695855
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.3407760, upper bound: 808.5260438
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.3407760, upper bound: 808.3695855
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.3407760, upper bound: 808.5260438
IS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0657336, upper bound: 808.0657503
IS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0657335, upper bound: 808.1705309
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0660411, upper bound: 808.0666594
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0660348, upper bound: 808.2531693
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0662031, upper bound: 808.0803281
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0662028, upper bound: 808.0956454
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0762917, upper bound: 808.0890156
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0762943, upper bound: 808.2640592
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0664789, upper bound: 808.0667754
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0664789, upper bound: 808.2533698
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0664789, upper bound: 808.0667754
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.0664829, upper bound: 808.2533698
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.3125838, upper bound: 808.2838647
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.3287463, upper bound: 808.3258292
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.3407760, upper bound: 808.3416901
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 4, lower bound: -808.3407760, upper bound: 808.3416901

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -265.4413147, 234.5291290, -324.2659607, 274.9338379, -540.3751221, 558.7951050
1: -212.0245361, 227.3792267, -258.7916260, 268.2318726, -480.2563477, 486.1707458
2: -308.5409546, 248.7822266, -374.0178223, 293.8462524, -602.3872070, 622.8000488
3: -124.6852646, 307.7318726, -149.7872467, 370.0194397, -494.7046509, 457.5191040
4: -343.8453064, 246.8465729, -417.8160400, 289.9113159, -633.7565918, 664.6624756

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8604675, upper bound: 806.2159515
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8604706, upper bound: 806.2167483
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -368.1410522, 308.8850098, -333.6746216, 280.5726624, -648.7137451, 642.5596313
1: -294.9066162, 299.8633118, -266.1934204, 273.5692749, -568.4757080, 566.0567627
2: -427.6830444, 327.7184753, -385.0966797, 299.6878357, -727.3707275, 712.8151855
3: -166.5783997, 420.1256409, -152.7088623, 379.5434570, -546.1218262, 572.8344727
4: -475.9859619, 324.2613831, -430.5206604, 295.3512878, -771.3371582, 754.7817993

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 33

Time for candidate selection: 3.59 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1713726, upper bound: 803.8251074
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1635231, upper bound: 804.3024402
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1635231, upper bound: 804.3024402
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -368.1410522, 308.8850098, -319.2808838, 270.4783630, -638.6193848, 628.1657715
1: -294.9066162, 299.8633118, -254.7389526, 263.8089905, -558.7155762, 554.6022949
2: -427.6830444, 327.7184753, -368.2485962, 288.9743652, -716.6573486, 695.9670410
3: -166.5783997, 420.1256409, -147.6340485, 363.9794922, -530.5578613, 567.7596436
4: -475.9859619, 324.2613831, -411.5106201, 285.0905762, -761.0764160, 735.7719727

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 33

Time for candidate selection: 3.74 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1710805, upper bound: 804.9907591
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 10

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1637174, upper bound: 806.2163487
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1637174, upper bound: 806.2163487
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -421.8020935, 353.0725708, -246.9592590, 215.8045654, -637.6066895, 600.0316162
1: -337.9725342, 342.4955139, -196.0946503, 210.8172607, -548.7897949, 538.5901489
2: -490.7892761, 374.1408997, -282.5487671, 231.6606140, -722.4498291, 656.6895752
3: -190.8523712, 481.2301636, -117.9779205, 282.5847473, -473.4371338, 599.2080688
4: -546.9939575, 370.5241394, -317.0406494, 228.7619324, -775.7558594, 687.5647583

Time for backsubstitution: 1.82 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=1011.34521484375
rel_dist={4: [-809.0067385931752, 809.0067385931752]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4689778, upper bound: 808.2251898
time: 0.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0037620, upper bound: 809.0037626
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.42 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.42
Output dim: 4, lower bound: -806.4689778, upper bound: 808.2251898
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.42
Output dim: 4, lower bound: -809.0037620, upper bound: 809.0037626

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -355.7917175, 298.0696411, -448.3735046, 376.2344360, -732.0261230, 746.4429932
1: -284.4114380, 290.6557922, -359.5960388, 364.8071289, -649.2185669, 650.2517700
2: -411.3229675, 317.9904480, -522.5211792, 398.0231934, -809.3461304, 840.5115967
3: -162.9068451, 405.6134033, -203.0482941, 512.1327515, -675.0396118, 608.6616821
4: -458.9186401, 313.6753845, -582.0831299, 394.3340454, -853.2526855, 895.7585449

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4683685, upper bound: 806.4683685
time: 0.70 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4683685, upper bound: 808.2251898
time: 0.68 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -462.0712891, 386.1998291, -841.2568359, 842.7302246
1: -364.9006348, 369.0178833, -370.5399170, 374.4891357, -739.3897705, 739.5578003
2: -530.4962158, 402.6548462, -538.6206055, 408.6528015, -939.1489258, 941.2754517
3: -205.3341675, 519.1251221, -208.4161530, 527.0053711, -732.3394775, 727.5412598
4: -590.9782715, 398.9001770, -599.9936523, 404.6311340, -995.6093750, 998.8937378

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2251898, upper bound: 806.4689778
time: 1.01 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2251898, upper bound: 806.4689778
time: 0.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.54 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.54
Output dim: 4, lower bound: -806.4683685, upper bound: 806.4683685
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 4, lower bound: -806.4683685, upper bound: 808.2251898
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 4, lower bound: -808.2251898, upper bound: 806.4689778
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 4, lower bound: -808.2251898, upper bound: 806.4689778

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -355.7917175, 298.0696411, -455.0569763, 380.6590271, -736.4506836, 753.1265869
1: -284.4114380, 290.6557922, -364.9006348, 369.0178833, -653.4293213, 655.5563965
2: -411.3229675, 317.9904480, -530.4962158, 402.6548462, -813.9777832, 848.4865723
3: -162.9068451, 405.6134033, -205.3341675, 519.1251221, -682.0319824, 610.9474487
4: -458.9186401, 313.6753845, -590.9782715, 398.9001770, -857.8187256, 904.6536865

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4197667, upper bound: 807.8053494
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4682934, upper bound: 808.1404324
time: 0.69 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -355.7917175, 298.0696411, -753.1265869, 736.4506836
1: -364.9006348, 369.0178833, -284.4114380, 290.6557922, -655.5563965, 653.4293213
2: -530.4962158, 402.6548462, -411.3229675, 317.9904480, -848.4865723, 813.9777832
3: -205.3341675, 519.1251221, -162.9068451, 405.6134033, -610.9473877, 682.0319824
4: -590.9782715, 398.9001770, -458.9186401, 313.6753845, -904.6536865, 857.8186646

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2251373, upper bound: 806.4688829
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1404305, upper bound: 806.4688297
time: 0.66 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -455.0569763, 380.6590271, -835.7160034, 835.7160034
1: -364.9006348, 369.0178833, -364.9006348, 369.0178833, -733.9185181, 733.9185181
2: -530.4962158, 402.6548462, -530.4962158, 402.6548462, -933.1510620, 933.1510620
3: -205.3341675, 519.1251221, -205.3341675, 519.1251221, -724.4591675, 724.4591675
4: -590.9782715, 398.9001770, -590.9782715, 398.9001770, -989.8784180, 989.8782959

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2251415, upper bound: 808.5813317
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1404324, upper bound: 808.5812559
time: 0.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.17 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 4, lower bound: -806.4197667, upper bound: 807.8053494
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 4, lower bound: -806.4682934, upper bound: 808.1404324
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 4, lower bound: -808.2251373, upper bound: 806.4688829
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 4, lower bound: -808.1404305, upper bound: 806.4688297
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 4, lower bound: -808.2251415, upper bound: 808.5813317
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 4, lower bound: -808.1404324, upper bound: 808.5812559

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -341.3818359, 288.1004028, -451.9305115, 377.8612061, -719.2430420, 740.0308838
1: -272.6688843, 281.0004883, -362.3603210, 366.3541870, -639.0228882, 643.3607178
2: -394.2149963, 307.6066589, -526.7690430, 399.7658691, -793.9808350, 834.3757324
3: -157.2244263, 389.4662170, -203.9234619, 515.4939575, -672.7183838, 593.3896484
4: -440.2426147, 303.5801392, -586.8534546, 396.0046997, -836.2473145, 890.4334717

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4202958, upper bound: 807.8053469
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4202958, upper bound: 807.8053494
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -346.6198730, 290.6638184, -452.2779236, 378.3785706, -724.9984131, 742.9416504
1: -276.9523926, 283.5098572, -362.6348267, 366.8122559, -643.7646484, 646.1445923
2: -400.2306824, 310.2190247, -527.1290283, 400.2485352, -800.4790649, 837.3480225
3: -158.6000671, 395.0196533, -204.1307068, 515.9241333, -674.5241699, 599.1500854
4: -446.5895996, 306.0231628, -587.2490234, 396.5334167, -843.1229858, 893.2721558

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4682714, upper bound: 808.1404305
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4688297, upper bound: 808.1404324
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -353.5406799, 296.1216736, -733.8940430, 720.9818115
1: -350.6974182, 356.2439270, -282.5860291, 288.7923889, -639.4898071, 638.8299561
2: -509.7792053, 388.8671265, -408.6560364, 315.9415588, -825.7207642, 797.5231934
3: -198.0632477, 499.4380493, -161.9555817, 403.0471497, -601.1104126, 661.3936157
4: -568.3402710, 385.4368286, -455.9725037, 311.6437988, -879.9840088, 841.4093018

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8053469, upper bound: 806.4202958
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8053469, upper bound: 806.4688297
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -353.5167847, 296.2268066, -741.1162720, 725.8076172
1: -356.6080017, 360.9292603, -282.5591736, 288.8795776, -645.4874878, 643.4884033
2: -518.1353760, 393.8309326, -408.5728455, 316.0592651, -834.1946411, 802.4038086
3: -200.9437408, 507.3840332, -161.8381653, 402.9840698, -603.9277344, 669.2221680
4: -577.2812500, 390.2192383, -455.8613586, 311.7717285, -889.0529785, 846.0805664

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0390655, upper bound: 806.3710522
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8053469, upper bound: 806.4202958
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8053386, upper bound: 806.4688297
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -451.9305115, 377.8612061, -815.6336060, 819.3717651
1: -350.6974182, 356.2439270, -362.3603210, 366.3541870, -717.0516357, 718.6041870
2: -509.7792053, 388.8671265, -526.7690430, 399.7658691, -909.5450439, 915.6361084
3: -198.0632477, 499.4380493, -203.9234619, 515.4939575, -713.5571899, 703.3613892
4: -568.3402710, 385.4368286, -586.8534546, 396.0046997, -964.3449707, 972.2902832

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0020427, upper bound: 808.2537522
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -452.2779236, 378.3785706, -823.2680664, 824.5688477
1: -356.6080017, 360.9292603, -362.6348267, 366.8122559, -723.4202881, 723.5640259
2: -518.1353760, 393.8309326, -527.1290283, 400.2485352, -918.3838501, 920.9599609
3: -200.9437408, 507.3840332, -204.1307068, 515.9241333, -716.8678589, 711.5146484
4: -577.2812500, 390.2192383, -587.2490234, 396.5334167, -973.8146362, 977.4682617

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5804876, upper bound: 808.2537942
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5813265, upper bound: 808.5812322
time: 1.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.29 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 4, lower bound: -806.4202958, upper bound: 807.8053469
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 4, lower bound: -806.4202958, upper bound: 807.8053494
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 4, lower bound: -806.4682714, upper bound: 808.1404305
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 4, lower bound: -806.4688297, upper bound: 808.1404324
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 4, lower bound: -807.8053469, upper bound: 806.4202958
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 4, lower bound: -807.8053469, upper bound: 806.4688297
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 4, lower bound: -807.8053469, upper bound: 806.4202958
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 4, lower bound: -807.8053386, upper bound: 806.4688297
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 4, lower bound: -808.5814650, upper bound: 808.5811934
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 4, lower bound: -808.5804876, upper bound: 808.2537942
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 4, lower bound: -808.5813265, upper bound: 808.5812322

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -341.3818359, 288.1004028, -437.7723999, 367.4412842, -708.8230591, 725.8727417
1: -272.6688843, 281.0004883, -350.6974182, 356.2439270, -628.9125977, 631.6978760
2: -394.2149963, 307.6066589, -509.7792053, 388.8671265, -783.0821533, 817.3858643
3: -157.2244263, 389.4662170, -198.0632477, 499.4380493, -656.6624756, 587.5294800
4: -440.2426147, 303.5801392, -568.3402710, 385.4368286, -825.6794434, 871.9202881

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.9122469, upper bound: 806.3943827
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2179647, upper bound: 807.6027576
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -341.3818359, 288.1004028, -444.8894653, 372.2908936, -713.6726685, 732.9898682
1: -272.6688843, 281.0004883, -356.6080017, 360.9292603, -633.5980835, 637.6085205
2: -394.2149963, 307.6066589, -518.1353760, 393.8309326, -788.0458984, 825.7420654
3: -157.2244263, 389.4662170, -200.9437408, 507.3840332, -664.6084595, 590.4099731
4: -440.2426147, 303.5801392, -577.2812500, 390.2192383, -830.4617920, 880.8613281

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.9122469, upper bound: 806.3943827
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2179647, upper bound: 807.6027596
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -346.6198730, 290.6638184, -437.7723999, 367.4412842, -714.0611572, 728.4360962
1: -276.9523926, 283.5098572, -350.6974182, 356.2439270, -633.1962891, 634.2072754
2: -400.2306824, 310.2190247, -509.7792053, 388.8671265, -789.0977173, 819.9982300
3: -158.6000671, 395.0196533, -198.0632477, 499.4380493, -658.0380249, 593.0827637
4: -446.5895996, 306.0231628, -568.3402710, 385.4368286, -832.0264282, 874.3634033

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8377067, upper bound: 807.5457232
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2179647, upper bound: 807.9628586
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -346.6198730, 290.6638184, -444.8894653, 372.2908936, -718.9107666, 735.5532837
1: -276.9523926, 283.5098572, -356.6080017, 360.9292603, -637.8815918, 640.1177979
2: -400.2306824, 310.2190247, -518.1353760, 393.8309326, -794.0615845, 828.3543701
3: -158.6000671, 395.0196533, -200.9437408, 507.3840332, -665.9840698, 595.9632568
4: -446.5895996, 306.0231628, -577.2812500, 390.2192383, -836.8088379, 883.3044434

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8377067, upper bound: 807.7354464
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2179647, upper bound: 807.9628608
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -341.3818359, 288.1004028, -725.8728027, 708.8230591
1: -350.6974182, 356.2439270, -272.6688843, 281.0004883, -631.6978760, 628.9125977
2: -509.7792053, 388.8671265, -394.2149963, 307.6066589, -817.3858643, 783.0821533
3: -198.0632477, 499.4380493, -157.2244263, 389.4662170, -587.5294800, 656.6624756
4: -568.3402710, 385.4368286, -440.2426147, 303.5801392, -871.9202881, 825.6794434

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.0766716, upper bound: 805.0107161
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 33

Time for candidate selection: 4.95 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1086270, upper bound: 805.2933963
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -803.7374135, upper bound: 805.2909212
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -346.6198730, 290.6638184, -728.4360352, 714.0611572
1: -350.6974182, 356.2439270, -276.9523926, 283.5098572, -634.2072144, 633.1962280
2: -509.7792053, 388.8671265, -400.2306824, 310.2190247, -819.9982300, 789.0977783
3: -198.0632477, 499.4380493, -158.6000671, 395.0196533, -593.0827637, 658.0380249
4: -568.3402710, 385.4368286, -446.5895996, 306.0231628, -874.3634033, 832.0264282

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.0766716, upper bound: 805.6747430
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 33

Time for candidate selection: 4.97 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1086270, upper bound: 805.4766120
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -803.7374135, upper bound: 805.4743648
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -341.3818359, 288.1004028, -732.9898682, 713.6727295
1: -356.6080017, 360.9292603, -272.6688843, 281.0004883, -637.6085205, 633.5980835
2: -518.1353760, 393.8309326, -394.2149963, 307.6066589, -825.7420654, 788.0458984
3: -200.9437408, 507.3840332, -157.2244263, 389.4662170, -590.4099731, 664.6084595
4: -577.2812500, 390.2192383, -440.2426147, 303.5801392, -880.8613281, 830.4617920

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.5804670, upper bound: 806.4194728
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8053469, upper bound: 806.4201796
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -346.6198730, 290.6638184, -735.5532227, 718.9107056
1: -356.6080017, 360.9292603, -276.9523926, 283.5098572, -640.1178589, 637.8816528
2: -518.1353760, 393.8309326, -400.2306824, 310.2190247, -828.3543701, 794.0615845
3: -200.9437408, 507.3840332, -158.6000671, 395.0196533, -595.9631958, 665.9840698
4: -577.2812500, 390.2192383, -446.5895996, 306.0231628, -883.3044434, 836.8088379

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.5804670, upper bound: 806.4681782
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8053469, upper bound: 806.4686174
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -437.7723999, 367.4412842, -805.2136230, 805.2136230
1: -350.6974182, 356.2439270, -350.6974182, 356.2439270, -706.9413452, 706.9413452
2: -509.7792053, 388.8671265, -509.7792053, 388.8671265, -898.6461792, 898.6463013
3: -198.0632477, 499.4380493, -198.0632477, 499.4380493, -697.5012817, 697.5012817
4: -568.3402710, 385.4368286, -568.3402710, 385.4368286, -953.7770996, 953.7770996

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8216420, upper bound: 808.4947800
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9354636, upper bound: 808.3421237
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9893061, upper bound: 808.3422657
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -444.8894653, 372.2908936, -810.0632935, 812.3307495
1: -350.6974182, 356.2439270, -356.6080017, 360.9292603, -711.6267090, 712.8518677
2: -509.7792053, 388.8671265, -518.1353760, 393.8309326, -903.6101074, 907.0024414
3: -198.0632477, 499.4380493, -200.9437408, 507.3840332, -705.4472656, 700.3817139
4: -568.3402710, 385.4368286, -577.2812500, 390.2192383, -958.5594482, 962.7180786

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9354636, upper bound: 808.3421237
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9893061, upper bound: 808.3422657
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -407.8981323, 343.9847717, -788.8742676, 780.1890259
1: -356.6080017, 360.9292603, -326.8345947, 333.2190247, -689.8269043, 687.7638550
2: -518.1353760, 393.8309326, -475.0801392, 363.6651611, -881.8005371, 868.9110718
3: -200.9437408, 507.3840332, -185.2439270, 465.3120728, -666.2557983, 692.6279297
4: -577.2812500, 390.2192383, -529.7960815, 360.7339478, -938.0151978, 920.0151978

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532139, upper bound: 808.2531362
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532139, upper bound: 808.2537942
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -426.9790649, 356.5452271, -706.7479248, 572.6456909, -995.6341553, 1063.2930908
1: -342.0051270, 345.6569214, -568.3411255, 556.1414795, -894.4108887, 913.9980469
2: -496.5918579, 377.4574585, -822.4509277, 605.5181885, -1098.0992432, 1199.9084473
3: -192.9383545, 486.0449524, -312.2856750, 795.4329224, -988.3712769, 794.6715698
4: -553.4179077, 374.0183411, -914.1930542, 598.5933228, -1148.2375488, 1288.2114258

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2535787, upper bound: 808.2640338
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2535787, upper bound: 808.5812322
time: 0.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.25 seconds
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.25
Output dim: 4, lower bound: -804.9122469, upper bound: 806.3943827
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -806.2179647, upper bound: 807.6027576
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.25
Output dim: 4, lower bound: -804.9122469, upper bound: 806.3943827
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -806.2179647, upper bound: 807.6027596
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -805.8377067, upper bound: 807.5457232
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -806.2179647, upper bound: 807.9628586
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -805.8377067, upper bound: 807.7354464
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -806.2179647, upper bound: 807.9628608
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -808.1086270, upper bound: 805.2933963
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.25
Output dim: 4, lower bound: -803.7374135, upper bound: 805.2909212
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -808.1086270, upper bound: 805.4766120
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.25
Output dim: 4, lower bound: -803.7374135, upper bound: 805.4743648
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.25
Output dim: 4, lower bound: -806.5804670, upper bound: 806.4194728
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -807.8053469, upper bound: 806.4201796
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.25
Output dim: 4, lower bound: -806.5804670, upper bound: 806.4681782
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -807.8053469, upper bound: 806.4686174
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -808.9354636, upper bound: 808.3421237
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -808.9893061, upper bound: 808.3422657
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -808.9354636, upper bound: 808.3421237
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -808.9893061, upper bound: 808.3422657
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -808.2532139, upper bound: 808.2531362
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -808.2532139, upper bound: 808.2537942
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -808.2535787, upper bound: 808.2640338
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 4, lower bound: -808.2535787, upper bound: 808.5812322

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -332.7971191, 281.7926941, -435.4188843, 365.4100342, -698.2070312, 717.2115479
1: -265.7110901, 274.8519287, -348.7833862, 354.3109131, -620.0219727, 623.6350098
2: -384.1240845, 300.9845886, -506.9671631, 386.7787476, -770.9027710, 807.9517822
3: -153.6954041, 379.7464600, -197.0967712, 496.7078552, -650.4032593, 576.8432007
4: -429.0441895, 297.0279236, -565.2241821, 383.3228455, -812.3668823, 862.2520752

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.1551196, upper bound: 805.8783954
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2182022, upper bound: 808.1220596
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -332.7971191, 281.7926941, -442.5560303, 370.2713623, -703.0684814, 724.3487549
1: -265.7110901, 274.8519287, -354.7106934, 359.0053101, -624.7164307, 629.5626221
2: -384.1240845, 300.9845886, -515.3446045, 391.8888855, -776.0128784, 816.3292236
3: -153.6954041, 379.7464600, -199.9835968, 504.6768494, -658.3722534, 579.7300415
4: -429.0441895, 297.0279236, -574.1887817, 388.1746826, -817.2187500, 871.2166748

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2167274, upper bound: 806.1217334
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2178334, upper bound: 807.6027596
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -243.7677917, 213.4292908, -397.9322510, 336.0661011, -579.8338623, 611.3615723
1: -193.7469482, 208.7287445, -318.4071655, 325.9954834, -519.7423706, 527.1359253
2: -278.9052429, 229.1844482, -462.4198608, 356.3890991, -635.2942505, 691.6041870
3: -116.8389282, 279.3439636, -181.8160095, 453.9105530, -570.7494507, 461.1598511
4: -312.4699707, 226.3032074, -515.9180298, 353.1409607, -665.6108398, 742.2211914

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.5938017, upper bound: 807.0308909
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.5938552, upper bound: 807.3443472
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -339.1815186, 285.2177429, -435.4188843, 365.4100342, -704.5915527, 720.6365967
1: -270.9281616, 278.1785583, -348.7833862, 354.3109131, -625.2390747, 626.9617920
2: -391.5458069, 304.4880066, -506.9671631, 386.7787476, -778.3245239, 811.4552002
3: -155.5240631, 386.5900574, -197.0967712, 496.7078552, -652.2318726, 583.6865845
4: -436.9592896, 300.3887939, -565.2241821, 383.3228455, -820.2821045, 865.6129761

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7840665, upper bound: 807.3084419
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7843643, upper bound: 807.9412979
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -243.7677917, 213.4292908, -405.9140015, 341.7361450, -585.5039062, 619.3432007
1: -193.7469482, 208.7287445, -324.9971008, 331.4360352, -525.1829834, 533.7257690
2: -278.9052429, 229.1844482, -471.7902832, 362.1820374, -641.0870972, 700.9747314
3: -116.8389282, 279.3439636, -185.0549774, 462.8110657, -579.6499634, 464.3989258
4: -312.4699707, 226.3032074, -526.0027466, 358.7987061, -671.2686157, 752.3059692

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8377130, upper bound: 807.7354458
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8377026, upper bound: 807.7173931
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -339.1815186, 285.2177429, -442.5560303, 370.2713623, -709.4528809, 727.7738037
1: -270.9281616, 278.1785583, -354.7106934, 359.0053101, -629.9334717, 632.8892822
2: -391.5458069, 304.4880066, -515.3446045, 391.8888855, -783.4346924, 819.8326416
3: -155.5240631, 386.5900574, -199.9835968, 504.6768494, -660.2009277, 586.5734863
4: -436.9592896, 300.3887939, -574.1887817, 388.1746826, -825.1339722, 874.5775757

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.0483181, upper bound: 807.8398506
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2178334, upper bound: 807.9628608
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -415.0517273, 349.3062744, -341.3818359, 288.1004028, -703.1520996, 690.6880493
1: -332.2945557, 338.7788391, -272.6688843, 281.0004883, -613.2950439, 611.4475098
2: -483.1736450, 369.8584900, -394.2149963, 307.6066589, -790.7802734, 764.0734863
3: -188.1912231, 474.1362305, -157.2244263, 389.4662170, -577.6574707, 631.3606567
4: -538.9747314, 366.5451660, -440.2426147, 303.5801392, -842.5548096, 806.7877808

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7948115, upper bound: 804.8267702
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1022246, upper bound: 805.2933791
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -415.0517273, 349.3062744, -346.6198730, 290.6638184, -705.7153931, 695.9261475
1: -332.2945557, 338.7788391, -276.9523926, 283.5098572, -615.8044434, 615.7311401
2: -483.1736450, 369.8584900, -400.2306824, 310.2190247, -793.3927002, 770.0891724
3: -188.1912231, 474.1362305, -158.6000671, 395.0196533, -583.2106934, 632.7362671
4: -538.9747314, 366.5451660, -446.5895996, 306.0231628, -844.9979248, 813.1347656

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5444392, upper bound: 805.3718656
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1029859, upper bound: 805.4765747
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -436.6898499, 365.1800842, -340.3303223, 287.2544556, -723.9443359, 705.5103760
1: -349.9454956, 354.0716248, -271.8060913, 280.1850281, -630.1304932, 625.8776855
2: -508.4296265, 386.5229187, -392.9669495, 306.7225342, -815.1521606, 779.4898682
3: -197.2324371, 497.9155884, -156.8147888, 388.2829895, -585.5153809, 654.7303467
4: -566.5614014, 382.8739319, -438.8776245, 302.7061768, -869.2675781, 821.7514648

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.3943827, upper bound: 804.9122259
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6027576, upper bound: 806.2178334
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -436.6898499, 365.1800842, -344.8656311, 289.2525940, -725.9423828, 710.0457153
1: -349.9454956, 354.0716248, -275.5284729, 282.1466064, -632.0920410, 629.6000977
2: -508.4296265, 386.5229187, -398.1666260, 308.7391357, -817.1687622, 784.6895142
3: -197.2324371, 497.9155884, -157.8805542, 393.0510559, -590.2835083, 655.7960815
4: -566.5614014, 382.8739319, -444.3177185, 304.5546570, -871.1160889, 827.1916504

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9729030, upper bound: 805.8371367
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6027576, upper bound: 806.2178334
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -362.6666565, 305.9226074, -412.3594360, 346.6170349, -709.2836914, 718.2820435
1: -290.2025757, 296.8375549, -330.1300049, 336.1081238, -626.3106689, 626.9675293
2: -421.4447021, 324.4236450, -479.7531738, 367.0744934, -788.5191650, 804.1768188
3: -164.7423096, 413.9035950, -186.8136444, 470.3801270, -635.1224365, 600.7172241
4: -469.5854187, 321.2560120, -534.8810425, 363.8028564, -833.3883057, 856.1370239

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9346316, upper bound: 808.8214256
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8843843, upper bound: 808.2829668
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.7131161, upper bound: 808.9583892
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 33

Time for candidate selection: 6.66 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8607541, upper bound: 808.9888223
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8606085, upper bound: 808.8516993
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -428.7608032, 359.9247742, -436.4138794, 366.3216248, -795.0822754, 796.3385620
1: -343.3657837, 348.9992065, -349.5941772, 355.1638489, -698.5296021, 698.5933838
2: -499.1161499, 381.0668945, -508.1755676, 387.7037659, -886.8198853, 889.2424316
3: -194.1269379, 489.1353455, -197.4744110, 497.8939209, -692.0208740, 686.6096191
4: -556.5690918, 377.6802368, -566.5706177, 384.2795105, -940.8486328, 944.2508545

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9885938, upper bound: 808.8212515
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9886953, upper bound: 808.9886804
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -362.6666565, 305.9226074, -420.7832642, 352.3573303, -715.0239868, 726.7057495
1: -290.2025757, 296.8375549, -337.1024780, 341.6651917, -631.8677979, 633.9400635
2: -421.4447021, 324.4236450, -489.6412964, 373.0296936, -794.4743042, 814.0648804
3: -164.7423096, 413.9035950, -190.1598663, 479.7093811, -644.4516602, 604.0634766
4: -469.5854187, 321.2560120, -545.4690552, 369.4468689, -839.0322876, 866.7250977

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9345861, upper bound: 808.2534778
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8686752, upper bound: 808.2842093
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8690350, upper bound: 808.3260304
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -428.7608032, 359.9247742, -443.4935608, 371.1280823, -799.8887329, 803.4183350
1: -343.3657837, 348.9992065, -355.4742126, 359.8077087, -703.1734619, 704.4733887
2: -499.1161499, 381.0668945, -516.4859009, 392.6305542, -891.7467041, 897.5527954
3: -194.1269379, 489.1353455, -200.3344269, 505.7943726, -699.9213257, 689.4696045
4: -556.5690918, 377.6802368, -575.4609375, 389.0164185, -945.5855103, 953.1411133

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9885003, upper bound: 808.2533275
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9892826, upper bound: 808.3422657
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9892826, upper bound: 808.3422657
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -400.8190918, 338.1311340, -407.8981323, 343.9847717, -744.8038330, 746.0292969
1: -321.0563354, 327.5602722, -326.8345947, 333.2190247, -654.2752075, 654.3947754
2: -466.4544678, 357.4834900, -475.0801392, 363.6651611, -830.1196289, 832.5635986
3: -182.1852417, 457.0790405, -185.2439270, 465.3120728, -647.4973145, 642.3228149
4: -520.2389526, 354.6555786, -529.7960815, 360.7339478, -880.9728394, 884.4514771

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2531886, upper bound: 808.0666448
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532577, upper bound: 808.2530238
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -681.0608521, 552.2008667, -407.8981323, 343.9847717, -1025.0456543, 956.6705933
1: -547.5318604, 536.2175293, -326.8345947, 333.2190247, -880.7507935, 859.7690430
2: -792.0316162, 583.9186401, -475.0801392, 363.6651611, -1155.6965332, 1055.4998779
3: -301.0640259, 766.3463135, -185.2439270, 465.3120728, -762.9222412, 951.5901489
4: -880.5621338, 577.2903442, -529.7960815, 360.7339478, -1241.2961426, 1103.8693848

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2531886, upper bound: 808.0667563
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532577, upper bound: 808.2532133
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -398.6757812, 336.2539673, -703.8887329, 570.3380127, -965.1093750, 1040.1427002
1: -319.2977600, 325.5302124, -566.0261841, 553.8973389, -869.5313110, 891.5563354
2: -463.9438477, 355.1840210, -819.0871582, 603.0971680, -1062.9986572, 1174.2712402
3: -181.0929260, 454.5400085, -311.0597229, 792.2706299, -973.3635254, 761.8509521
4: -517.4816895, 352.5979614, -910.4666138, 596.2365112, -1109.9174805, 1263.0645752

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2531809, upper bound: 808.2640338
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2531809, upper bound: 808.2640338
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -683.3704834, 554.1118774, -706.7766113, 572.6698608, -1246.3299561, 1251.6939697
1: -549.4003906, 538.0770874, -568.3643188, 556.1651611, -1097.7989502, 1099.0661621
2: -794.7464600, 585.9254150, -822.4848633, 605.5437622, -1391.3142090, 1400.0046387
3: -302.0794983, 768.9109497, -312.2984619, 795.4650269, -1092.3564453, 1075.6605225
4: -883.5769653, 579.2427979, -914.2306519, 598.6181641, -1472.4283447, 1484.3985596

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2531809, upper bound: 808.5812322
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2531809, upper bound: 808.5812322
time: 0.78 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.36 seconds
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 4, lower bound: -806.1551196, upper bound: 805.8783954
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -806.2182022, upper bound: 808.1220596
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 4, lower bound: -806.2167274, upper bound: 806.1217334
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -806.2178334, upper bound: 807.6027596
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 4, lower bound: -805.5938017, upper bound: 807.0308909
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 4, lower bound: -805.5938552, upper bound: 807.3443472
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 4, lower bound: -805.7840665, upper bound: 807.3084419
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -805.7843643, upper bound: 807.9412979
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -805.8377130, upper bound: 807.7354458
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -805.8377026, upper bound: 807.7173931
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -806.0483181, upper bound: 807.8398506
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -806.2178334, upper bound: 807.9628608
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 4, lower bound: -806.7948115, upper bound: 804.8267702
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.1022246, upper bound: 805.2933791
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -807.5444392, upper bound: 805.3718656
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.1029859, upper bound: 805.4765747
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 4, lower bound: -806.3943827, upper bound: 804.9122259
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -807.6027576, upper bound: 806.2178334
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 4, lower bound: -806.9729030, upper bound: 805.8371367
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -807.6027576, upper bound: 806.2178334
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.8607541, upper bound: 808.9888223
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.8606085, upper bound: 808.8516993
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.9885938, upper bound: 808.8212515
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.9886953, upper bound: 808.9886804
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.8686752, upper bound: 808.2842093
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.8690350, upper bound: 808.3260304
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.9892826, upper bound: 808.3422657
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.9892826, upper bound: 808.3422657
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.2531886, upper bound: 808.0666448
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.2532577, upper bound: 808.2530238
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.2531886, upper bound: 808.0667563
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.2532577, upper bound: 808.2532133
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.2531809, upper bound: 808.2640338
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.2531809, upper bound: 808.2640338
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.2531809, upper bound: 808.5812322
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 4, lower bound: -808.2531809, upper bound: 808.5812322

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -332.7971191, 281.7926941, -431.6472778, 362.1653137, -694.9624023, 713.4398804
1: -265.7110901, 274.8519287, -345.7006836, 351.1658325, -616.8769531, 620.5525513
2: -384.1240845, 300.9845886, -502.4071960, 383.3771667, -767.5010376, 803.3917847
3: -153.6954041, 379.7464600, -195.3830261, 492.2681580, -645.9635620, 575.1295166
4: -429.0441895, 297.0279236, -560.1879272, 379.9534302, -808.9974365, 857.2158203

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.0736592, upper bound: 807.2094854
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.0736592, upper bound: 808.1220599
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -331.3330994, 280.6216125, -434.4464111, 363.2101135, -694.5432129, 715.0679932
1: -264.5181274, 273.7242432, -348.1134033, 352.2248230, -616.7429199, 621.8376465
2: -382.3978271, 299.7616577, -505.7304077, 384.6291504, -767.0269775, 805.4920654
3: -153.1101379, 378.1066895, -196.2941742, 495.2881775, -648.3982544, 574.4008789
4: -427.1453552, 295.8182068, -563.5686035, 380.9730225, -808.1183472, 859.3868408

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9771565, upper bound: 807.5472560
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 26

Time for candidate selection: 4.90 seconds

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2177088, upper bound: 807.5784228
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.6228577, upper bound: 807.3951201
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.1691302, upper bound: 804.8815485
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.2910598, upper bound: 804.8806960
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -336.6150208, 283.3428955, -502.7843018, 418.1448059, -754.7598267, 786.1271973
1: -268.8470459, 276.3506470, -403.2431946, 405.4440308, -674.2910767, 679.5938721
2: -388.5213013, 302.4979248, -585.7991333, 442.4644165, -830.9856567, 888.2969360
3: -154.4554443, 383.6843262, -227.0589600, 572.3013916, -726.7568359, 610.7432251
4: -433.6389160, 298.4492188, -652.8253784, 438.9445801, -872.5834961, 951.2745972

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7564227, upper bound: 807.8676154
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7840578, upper bound: 807.9412979
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -226.4915771, 199.6800537, -338.6876221, 286.2373047, -512.7288818, 538.3676147
1: -179.8034821, 195.4608612, -270.8771057, 277.8439331, -457.6473999, 466.3378906
2: -258.6960449, 214.7748566, -392.5540466, 303.9434204, -562.6394653, 607.3289185
3: -109.1338882, 259.8230591, -154.7837524, 385.9906006, -495.1243896, 414.6068115
4: -289.9578552, 212.0751953, -437.2468262, 300.8225098, -590.7803345, 649.3220215

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.1406377, upper bound: 806.1424228
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.4057069, upper bound: 806.1424228
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -242.7099762, 212.6123199, -390.0116882, 328.3514404, -571.0614014, 602.6240234
1: -192.8807983, 207.9384155, -312.0198975, 318.5456848, -511.4264526, 519.9583130
2: -277.6422729, 228.3321381, -452.7027893, 348.3702087, -626.0124512, 681.0349121
3: -116.4037476, 278.1564941, -178.1473083, 444.3056030, -560.7093506, 456.3037109
4: -311.0840759, 225.4624634, -504.9284668, 344.9952698, -656.0792847, 730.3908691

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.6876182, upper bound: 807.6209202
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5675413, upper bound: 807.6034049
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -319.7500610, 269.4739685, -378.0057678, 316.5083618, -636.2584229, 647.4796143
1: -255.2561646, 263.0362854, -302.7718506, 307.0903320, -562.3464966, 565.8081055
2: -368.7505798, 288.0944214, -439.3327942, 335.5221558, -704.2727051, 727.4270630
3: -146.6914368, 364.4205933, -170.7598572, 430.7801514, -577.4715576, 535.1804199
4: -411.5258179, 284.1374512, -488.9616089, 332.1539307, -743.6797485, 773.0990601

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9357089, upper bound: 807.8293769
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.5562224, upper bound: 807.1165575
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7775956, upper bound: 807.8191695
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -337.3961792, 283.7915955, -434.4464111, 363.2101135, -700.6063232, 718.2379761
1: -269.4771118, 276.8007202, -348.1134033, 352.2248230, -621.7019043, 624.9141235
2: -389.4316711, 302.9927368, -505.7304077, 384.6291504, -774.0607910, 808.7230835
3: -154.7963409, 384.5790405, -196.2941742, 495.2881775, -650.0844727, 580.8732300
4: -434.6307068, 298.9100952, -563.5686035, 380.9730225, -815.6037598, 862.4786377

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.9383602, upper bound: 806.9378098
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9498298, upper bound: 807.9304776
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2172850, upper bound: 807.9583636
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -412.7009277, 347.3033752, -332.7971191, 281.7926941, -694.4935303, 680.1004639
1: -330.3798218, 336.9181824, -265.7110901, 274.8519287, -605.2316895, 602.6292725
2: -480.3560791, 367.9341431, -384.1240845, 300.9845886, -781.3406982, 752.0579834
3: -187.2425537, 471.3845825, -153.6954041, 379.7464600, -566.9890137, 625.0799561
4: -535.8590698, 364.6180725, -429.0441895, 297.0279236, -832.8869629, 793.6621704

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8105194, upper bound: 804.0588171
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0831791, upper bound: 805.2933729
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -375.0715027, 318.5820923, -243.7677917, 213.4292908, -588.5007935, 562.3498535
1: -299.8727417, 309.1310730, -193.7469482, 208.7287445, -508.6014404, 502.8780212
2: -435.6372070, 337.9595642, -278.9052429, 229.1844482, -664.8216553, 616.8645630
3: -172.0508118, 428.6138000, -116.8389282, 279.3439636, -451.3947449, 545.4527588
4: -486.3757019, 334.9399109, -312.4699707, 226.3032074, -712.6788330, 647.4098511

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.0295230, upper bound: 805.3711505
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3431259, upper bound: 805.3713415
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -412.7009277, 347.3033752, -339.1815186, 285.2177429, -697.9186401, 686.4848633
1: -330.3798218, 336.9181824, -270.9281616, 278.1785583, -608.5583496, 607.8463135
2: -480.3560791, 367.9341431, -391.5458069, 304.4880066, -784.8440552, 759.4798584
3: -187.2425537, 471.3845825, -155.5240631, 386.5900574, -573.8324585, 626.9085693
4: -535.8590698, 364.6180725, -436.9592896, 300.3887939, -836.2478638, 801.5773926

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3057728, upper bound: 805.4757683
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9371575, upper bound: 805.4762402
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -434.4464111, 363.2101135, -331.3330994, 280.6216125, -715.0679321, 694.5432129
1: -348.1134033, 352.2248230, -264.5181274, 273.7242432, -621.8376465, 616.7429199
2: -505.7304077, 384.6291504, -382.3978271, 299.7616577, -805.4920654, 767.0269775
3: -196.2941742, 495.2881775, -153.1101379, 378.1066895, -574.4008789, 648.3981934
4: -563.5686035, 380.9730225, -427.1453552, 295.8182068, -859.3868408, 808.1183472

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.2022028, upper bound: 806.2166689
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6005589, upper bound: 806.2172850
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -434.4464111, 363.2101135, -337.3961792, 283.7915955, -718.2379150, 700.6063232
1: -348.1134033, 352.2248230, -269.4771118, 276.8007202, -624.9141235, 621.7019043
2: -505.7304077, 384.6291504, -389.4316711, 302.9927368, -808.7230835, 774.0607910
3: -196.2941742, 495.2881775, -154.7963409, 384.5790405, -580.8732300, 650.0844727
4: -563.5686035, 380.9730225, -434.6307068, 298.9100952, -862.4786987, 815.6037598

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7838111, upper bound: 805.7831679
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.4462567, upper bound: 806.2166689
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6005589, upper bound: 806.2172850
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -362.6666565, 305.9226074, -322.5835876, 280.8328857, -643.4994507, 628.5061646
1: -290.2025757, 296.8375549, -257.9532471, 272.3260193, -562.5285645, 554.7907715
2: -421.4447021, 324.4236450, -375.6186218, 298.2458496, -719.6905518, 700.0421143
3: -164.7423096, 413.9035950, -150.7577057, 371.0143738, -535.7567139, 564.6613159
4: -469.5854187, 321.2560120, -418.6109314, 295.5288391, -765.1142578, 739.8669434

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4590576, upper bound: 808.9875678
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8606171, upper bound: 808.9882473
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -362.6666565, 305.9226074, -397.3122253, 333.7330627, -696.3995972, 703.2348633
1: -290.2025757, 296.8375549, -317.8696594, 323.3340454, -613.5366211, 614.7072144
2: -421.4447021, 324.4236450, -461.7725830, 353.2521667, -774.6968384, 786.1962280
3: -164.7423096, 413.9035950, -179.8338928, 452.5745239, -617.3168335, 593.7374878
4: -469.5854187, 321.2560120, -515.0405884, 350.1349182, -819.7203369, 836.2966309

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4589665, upper bound: 808.8504472
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8605165, upper bound: 808.8512033
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -428.7608032, 359.9247742, -393.6322327, 333.1250305, -761.8858032, 753.5570068
1: -343.3657837, 348.9992065, -315.0977173, 322.7879028, -666.1535645, 664.0968628
2: -499.1161499, 381.0668945, -457.9815063, 352.4102783, -851.5263062, 839.0484009
3: -194.1269379, 489.1353455, -179.2218170, 448.9517822, -643.0787354, 668.3569336
4: -556.5690918, 377.6802368, -511.1630859, 349.6517944, -906.2208862, 888.8433228

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8209940, upper bound: 808.8200733
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8209940, upper bound: 808.8212515
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -409.3800964, 342.9571838, -672.0935669, 545.8280640, -950.4999390, 1015.0506592
1: -327.5885315, 332.5180359, -540.2768555, 530.1953125, -853.4888916, 872.7947998
2: -475.8901062, 363.3493347, -781.6875610, 577.3547974, -1048.4348145, 1145.0368652
3: -185.5083008, 466.1105957, -297.4951172, 756.7155762, -942.2238770, 759.6214600
4: -530.8421631, 360.1704407, -869.2905273, 570.7171631, -1096.9340820, 1229.4609375

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8212379, upper bound: 808.8365467
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8212379, upper bound: 808.9886804
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -329.1347351, 281.7643127, -288.8924255, 255.9548645, -585.0895996, 570.6567383
1: -263.1374512, 273.3742065, -230.7728729, 247.9276428, -511.0650940, 504.1470032
2: -382.1664429, 298.9670105, -335.8865967, 271.3745422, -653.5408936, 634.8536377
3: -151.3080597, 376.8701477, -136.7850647, 334.5797729, -485.8878174, 513.6552124
4: -426.0687256, 296.1392212, -374.7416992, 269.2900085, -695.3587646, 670.8807373

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4581953, upper bound: 808.2840668
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4581953, upper bound: 808.2842092
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -359.2873230, 303.3527527, -407.7199707, 341.5518799, -700.8390503, 711.0727539
1: -287.4934387, 294.4251404, -326.5824280, 331.4229431, -618.9163818, 621.0073853
2: -417.4676819, 321.8305054, -474.1162720, 362.1206055, -779.5882568, 795.9467773
3: -163.3672791, 410.2485657, -184.5193481, 464.9274902, -628.2947998, 594.7679443
4: -465.1645203, 318.6399231, -528.2318726, 358.5128174, -823.6773071, 846.8718262

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4585489, upper bound: 808.3258879
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4585488, upper bound: 808.3260304
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -428.7608032, 359.9247742, -380.1496887, 318.2424622, -747.0031738, 740.0744019
1: -343.3657837, 348.9992065, -304.5241394, 308.7597961, -652.1256104, 653.5233154
2: -499.1161499, 381.0668945, -441.9195862, 337.3076477, -836.4235840, 822.9864502
3: -194.1269379, 489.1353455, -171.6442566, 433.2800598, -627.4069824, 660.7793579
4: -556.5690918, 377.6802368, -491.8168945, 333.9521179, -890.5212402, 869.4971313

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8211359, upper bound: 808.3415124
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2830480, upper bound: 808.3322474
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9766031, upper bound: 808.3327348
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -428.7608032, 359.9247742, -436.6898499, 365.1800842, -793.9407959, 796.6146240
1: -343.3657837, 348.9992065, -349.9454956, 354.0716248, -697.4373779, 698.9446411
2: -499.1161499, 381.0668945, -508.4296265, 386.5229187, -885.6388550, 889.4965210
3: -194.1269379, 489.1353455, -197.2324371, 497.9155884, -692.0425415, 686.3676147
4: -556.5690918, 377.6802368, -566.5614014, 382.8739319, -939.4429932, 944.2415771

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8211359, upper bound: 808.3415124
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2830480, upper bound: 808.3322474
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9766031, upper bound: 808.3327348
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -376.2217407, 318.0363464, -354.5906372, 299.0041199, -675.2258301, 672.6268311
1: -301.1268616, 308.1643982, -284.0128784, 289.9843750, -591.1112061, 592.1771240
2: -437.3203125, 336.4246216, -412.1521301, 316.5235901, -753.8438721, 748.5767822
3: -171.2566223, 428.7483826, -160.8089142, 404.7148132, -575.9714355, 589.5573120
4: -487.8498535, 333.6655273, -459.0503845, 313.7569275, -801.6067505, 792.7159424

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9938485, upper bound: 806.8860390
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2324641, upper bound: 808.0497838
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -399.4912415, 337.0315857, -400.3497314, 337.3345337, -736.8256226, 737.3813477
1: -319.9780579, 326.4995422, -320.6986084, 326.8034973, -646.7815552, 647.1980591
2: -464.8884583, 356.3374634, -466.1497192, 356.7325134, -821.6209717, 822.4871826
3: -181.6050110, 455.5646667, -181.7738190, 456.5380554, -638.1429443, 637.3384399
4: -518.5121460, 353.5135803, -519.9400635, 353.8379211, -872.3500366, 873.4535522

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0667793, upper bound: 808.1704481
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0667793, upper bound: 808.2530238
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -653.5577393, 529.9961548, -354.5906372, 299.0041199, -952.5617676, 879.6788330
1: -525.2982788, 514.7621460, -284.0128784, 289.9843750, -815.2825928, 794.3287354
2: -759.7069092, 560.7020264, -412.1521301, 316.5235901, -1076.2304688, 967.9375000
3: -289.1292725, 734.7678223, -160.8089142, 404.7148132, -689.4351807, 895.5767212
4: -844.5901489, 554.3016968, -459.0503845, 313.7569275, -1158.3470459, 1008.5440674

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415676, upper bound: 808.0667563
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415676, upper bound: 808.0667563
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -680.0932007, 551.3455811, -400.3497314, 337.3345337, -1017.4276733, 947.8159180
1: -546.7465210, 535.3922729, -320.6986084, 326.8034973, -873.5500488, 852.4363403
2: -790.8845215, 583.0256958, -466.1497192, 356.7325134, -1147.6168213, 1045.0863037
3: -300.5981445, 765.2337036, -181.7738190, 456.5380554, -753.4655762, 947.0075073
4: -879.2940063, 576.3933105, -519.9400635, 353.8379211, -1233.1318359, 1092.5164795

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3416172, upper bound: 808.2532133
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3416172, upper bound: 808.2532133
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -398.6757812, 336.2539673, -687.8900146, 558.2171631, -952.7420044, 1024.1439209
1: -319.2977600, 325.5302124, -553.0617676, 542.3068848, -857.7525635, 878.5919189
2: -463.9438477, 355.1840210, -800.2419434, 590.4776611, -1050.1766357, 1155.4259033
3: -181.0929260, 454.5400085, -304.4592285, 774.4904785, -955.5833740, 755.0579224
4: -517.4816895, 352.5979614, -889.7783813, 583.7472534, -1097.1835938, 1242.3762207

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36

Time for candidate selection: 4.34 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6063010, upper bound: 808.1402854
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1888449, upper bound: 808.2002392
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -398.6757812, 336.2539673, -696.9815674, 564.6813354, -959.5231934, 1033.2355957
1: -319.2977600, 325.5302124, -560.4191284, 548.4302979, -864.1157227, 885.9493408
2: -463.9438477, 355.1840210, -810.7377930, 597.1450195, -1057.1127930, 1165.9216309
3: -181.0929260, 454.5400085, -308.0501099, 784.2127075, -965.3056641, 758.8701782
4: -517.4816895, 352.5979614, -901.2353516, 590.3754883, -1104.1187744, 1253.8331299

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36

Time for candidate selection: 4.39 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6062808, upper bound: 808.1402854
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1888449, upper bound: 808.2002333
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -683.3704834, 554.1118774, -690.5591431, 560.4059448, -1233.8239746, 1235.3820801
1: -549.4003906, 538.0770874, -555.2222290, 544.4357300, -1085.8851318, 1085.8304443
2: -794.7464600, 585.9254150, -803.3838501, 592.7733765, -1378.3461914, 1380.7603760
3: -302.0794983, 768.9109497, -305.6170349, 777.4598389, -1074.1958008, 1068.7886963
4: -883.5769653, 579.2427979, -893.2622681, 585.9860229, -1459.5716553, 1463.2124023

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5462555, upper bound: 808.3421341
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3415822, upper bound: 808.3415989
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -683.3704834, 554.1118774, -699.7241821, 566.8917847, -1240.6314697, 1244.2750244
1: -549.4003906, 538.0770874, -562.6397705, 550.5788574, -1092.2719727, 1093.0689697
2: -794.7464600, 585.9254150, -813.9642944, 599.4636841, -1385.3099365, 1391.0814209
3: -302.0794983, 768.9109497, -309.2238159, 787.2454834, -1083.9855957, 1072.6212158
4: -883.5769653, 579.2427979, -904.8098755, 592.6386719, -1466.5336914, 1474.5134277

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3800477, upper bound: 808.3290536
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3289645, upper bound: 808.3285997
time: 0.70 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.78 seconds
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.0736592, upper bound: 807.2094854
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.0736592, upper bound: 808.1220599
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.78
Output dim: 4, lower bound: -806.1691302, upper bound: 804.8815485
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.2910598, upper bound: 804.8806960
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.7564227, upper bound: 807.8676154
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.7840578, upper bound: 807.9412979
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.1406377, upper bound: 806.1424228
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.4057069, upper bound: 806.1424228
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.6876182, upper bound: 807.6209202
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.5675413, upper bound: 807.6034049
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.5562224, upper bound: 807.1165575
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.7775956, upper bound: 807.8191695
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.9498298, upper bound: 807.9304776
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -806.2172850, upper bound: 807.9583636
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.78
Output dim: 4, lower bound: -805.8105194, upper bound: 804.0588171
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.0831791, upper bound: 805.2933729
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.78
Output dim: 4, lower bound: -807.0295230, upper bound: 805.3711505
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.78
Output dim: 4, lower bound: -807.3431259, upper bound: 805.3713415
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.78
Output dim: 4, lower bound: -807.3057728, upper bound: 805.4757683
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -807.9371575, upper bound: 805.4762402
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.78
Output dim: 4, lower bound: -807.2022028, upper bound: 806.2166689
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -807.6005589, upper bound: 806.2172850
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -807.4462567, upper bound: 806.2166689
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -807.6005589, upper bound: 806.2172850
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.4590576, upper bound: 808.9875678
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.8606171, upper bound: 808.9882473
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.4589665, upper bound: 808.8504472
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.8605165, upper bound: 808.8512033
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.8209940, upper bound: 808.8200733
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.8209940, upper bound: 808.8212515
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.8212379, upper bound: 808.8365467
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.8212379, upper bound: 808.9886804
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.4581953, upper bound: 808.2840668
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.4581953, upper bound: 808.2842092
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.4585489, upper bound: 808.3258879
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.4585488, upper bound: 808.3260304
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.2830480, upper bound: 808.3322474
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.9766031, upper bound: 808.3327348
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.2830480, upper bound: 808.3322474
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.9766031, upper bound: 808.3327348
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.78
Output dim: 4, lower bound: -806.9938485, upper bound: 806.8860390
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.2324641, upper bound: 808.0497838
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.0667793, upper bound: 808.1704481
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.0667793, upper bound: 808.2530238
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.3415676, upper bound: 808.0667563
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.3415676, upper bound: 808.0667563
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.3416172, upper bound: 808.2532133
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.3416172, upper bound: 808.2532133
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -807.6063010, upper bound: 808.1402854
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.1888449, upper bound: 808.2002392
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -807.6062808, upper bound: 808.1402854
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.1888449, upper bound: 808.2002333
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.5462555, upper bound: 808.3421341
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.3415822, upper bound: 808.3415989
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.3800477, upper bound: 808.3290536
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.78
Output dim: 4, lower bound: -808.3289645, upper bound: 808.3285997

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -330.0129395, 279.4838257, -431.6472778, 362.1653137, -692.1782227, 711.1309814
1: -263.4294434, 272.6274414, -345.7006836, 351.1658325, -614.5952759, 618.3280640
2: -380.7520752, 298.5825806, -502.4071960, 383.3771667, -764.1292725, 800.9896851
3: -152.3882141, 376.5030823, -195.3830261, 492.2681580, -644.6563110, 571.8860474
4: -425.3255005, 294.6168213, -560.1879272, 379.9534302, -805.2788086, 854.8046265

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.0716322, upper bound: 805.2289945
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.0717054, upper bound: 807.8292656
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -206.2215271, 185.9427032, -466.6304321, 391.6851501, -597.9065552, 652.5730591
1: -164.0267792, 181.8007660, -373.9967041, 379.6623840, -543.6890259, 555.7974854
2: -237.0758209, 200.1469269, -543.3653564, 414.5085449, -651.5843506, 743.5122681
3: -100.9501343, 239.7610321, -211.9677734, 532.1130371, -633.0631714, 451.7288208
4: -265.3192749, 197.2737122, -605.8068237, 411.2140198, -676.5333252, 803.0805664

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 43

Time for candidate selection: 4.15 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.2778047, upper bound: 807.8663548
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7506497, upper bound: 807.8675856
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7506497, upper bound: 807.8244731
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -322.7984924, 273.0111389, -499.6339111, 415.5224915, -738.3208618, 772.6450195
1: -257.7910156, 266.5462646, -400.7223511, 402.9478455, -660.7388916, 667.2685547
2: -372.2712097, 291.8565063, -582.0803833, 439.7591553, -812.0303345, 873.9368896
3: -149.2075958, 368.9168396, -225.7565613, 568.7755737, -717.9831543, 594.6734009
4: -415.6865234, 287.8476868, -648.6975708, 436.2351990, -851.9217529, 936.5452271

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 43

Time for candidate selection: 3.99 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7676650, upper bound: 807.9257172
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7676650, upper bound: 807.8958573
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -128.0101929, 131.2864227, -344.1691589, 294.0625000, -422.0726929, 475.4555664
1: -100.9231110, 128.6269531, -274.8076782, 285.4152222, -386.3383179, 403.4346313
2: -145.0767365, 142.8504028, -398.4775085, 312.6930237, -457.7697754, 541.3278809
3: -71.9495850, 154.0731354, -159.6621857, 393.1130676, -465.0626526, 313.7353210
4: -163.6014404, 141.0533447, -444.8821411, 309.5040283, -473.1054382, 585.9354248

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5669805, upper bound: 807.4482260
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5669805, upper bound: 807.6034049
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -233.5189819, 206.3169708, -380.8139038, 320.7360229, -554.2550049, 587.1308594
1: -185.5336304, 201.9783173, -304.5343628, 311.2870789, -496.8207092, 506.5126953
2: -266.7876587, 221.9416962, -441.5845337, 340.6019287, -607.3895874, 663.5262451
3: -112.9596481, 268.6106873, -174.2135468, 433.7643738, -546.7239990, 442.8242188
4: -299.0774536, 219.1005249, -492.6452637, 337.1752930, -636.2527466, 711.7457886

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5669805, upper bound: 807.4482260
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5669805, upper bound: 807.6034049
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -384.4769592, 320.6025085, -372.4288330, 312.3259277, -696.8028564, 693.0311890
1: -307.5166016, 311.6820068, -298.2412109, 303.0081482, -610.5245972, 609.9231567
2: -444.3940735, 340.7566223, -432.7558289, 331.0642090, -775.4582520, 773.5124512
3: -175.8102264, 437.3827820, -168.4761353, 424.5022278, -600.3124390, 605.8588867
4: -495.7594910, 337.1976013, -481.7518311, 327.7804871, -823.5397949, 818.9492798

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7769139, upper bound: 807.3830836
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 33

Time for candidate selection: 5.05 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.4746228, upper bound: 807.8119058
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7642616, upper bound: 807.8191278
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7643473, upper bound: 807.8191278
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -207.4743347, 186.7742004, -398.0147705, 336.5316467, -544.0059814, 584.7889404
1: -165.0382233, 182.6228180, -318.6360168, 326.2489929, -491.2872009, 501.2588196
2: -238.5079651, 201.0530853, -463.0353088, 356.4313965, -594.9393311, 664.0883179
3: -101.4312057, 241.2066650, -181.5359497, 454.8426819, -556.2738647, 422.7425537
4: -266.8806152, 198.1386719, -516.1264648, 353.1224060, -620.0030518, 714.2650757

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7559964, upper bound: 806.8390906
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7561383, upper bound: 807.9115819
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -323.9132080, 273.6915894, -431.3274536, 360.6111755, -684.5242310, 705.0189819
1: -258.6922607, 267.2218628, -345.6059265, 349.7778625, -608.4700928, 612.8277588
2: -373.5930176, 292.5969849, -502.0266113, 382.0057068, -755.5987549, 794.6234741
3: -149.6821442, 370.1916504, -194.9356232, 491.7581787, -641.4403076, 565.1272583
4: -417.1346436, 288.5514832, -559.4561157, 378.3387756, -795.4733276, 848.0075073

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7835740, upper bound: 806.8391819
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7837161, upper bound: 807.9325593
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -408.8341064, 344.0358276, -332.7971191, 281.7926941, -690.6267090, 676.8329468
1: -327.2171631, 333.7428894, -265.7110901, 274.8519287, -602.0689087, 599.4539795
2: -475.6848755, 364.4886475, -384.1240845, 300.9845886, -776.6694336, 748.6127319
3: -185.4908447, 466.8360291, -153.6954041, 379.7464600, -565.2373047, 620.5314331
4: -530.7052002, 361.2162781, -429.0441895, 297.0279236, -827.7331543, 790.2603149

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.5976814, upper bound: 803.6419111
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.5976814, upper bound: 805.2933729
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -472.6323853, 394.5634155, -336.6150208, 283.3428955, -755.9752808, 731.1784668
1: -378.8103943, 382.6238098, -268.8470459, 276.3506470, -655.1610107, 651.4708252
2: -550.4449463, 417.4819336, -388.5213013, 302.4979248, -852.9428711, 806.0031738
3: -213.7846375, 538.5390625, -154.4554443, 383.6843262, -597.4689941, 692.9945068
4: -613.8167725, 414.2570801, -433.6389160, 298.4492188, -912.2659912, 847.8957520

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8663526, upper bound: 805.2778047
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9359920, upper bound: 804.6427494
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -422.9704285, 353.7397766, -328.1409607, 278.2336731, -701.2041016, 681.8806152
1: -338.8988953, 343.2080688, -261.9611816, 271.4588318, -610.3577271, 605.1690063
2: -492.1096497, 374.9461365, -378.6616516, 297.3044739, -789.4141235, 753.6076050
3: -191.2827454, 482.4552917, -151.8966827, 374.6951599, -565.9779053, 634.3519897
4: -548.4427490, 371.2631226, -423.0267334, 293.3731384, -841.8156738, 794.2897949

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5472560, upper bound: 805.9766842
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 26

Time for candidate selection: 5.16 seconds

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5784228, upper bound: 806.2172850
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3729847, upper bound: 805.6223722
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.1947533, upper bound: 806.1418696
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.1939149, upper bound: 805.2898744
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -299.8043213, 265.0198669, -303.4079895, 259.4006042, -559.2048340, 568.4277344
1: -239.5955658, 256.6430054, -242.0749969, 253.2018127, -492.7973633, 498.7180176
2: -348.8709106, 280.9627991, -349.8569336, 277.6175842, -626.4885254, 630.8195801
3: -141.9506378, 347.1585388, -141.1991882, 347.2369995, -489.1876221, 488.3576660
4: -389.3030090, 278.8648682, -390.7096863, 273.7729187, -663.0759277, 669.5745239

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0014141, upper bound: 805.9488433
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0014402, upper bound: 806.2166689
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -422.9704285, 353.7397766, -333.5852661, 280.9348145, -703.9052734, 687.3248291
1: -338.8988953, 343.2080688, -266.4122009, 274.0831909, -612.9820557, 609.6200562
2: -492.1096497, 374.9461365, -384.8757935, 300.0490417, -792.1585693, 759.8217773
3: -191.2827454, 482.4552917, -153.3569031, 380.4230347, -571.7057495, 635.8121948
4: -548.4427490, 371.2631226, -429.5957031, 295.9784851, -844.4208984, 800.8588257

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0014402, upper bound: 805.9488510
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0014402, upper bound: 806.2172850
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -238.8508759, 215.9740753, -307.4263306, 269.3284912, -508.1793518, 523.4003906
1: -190.3831024, 209.4111938, -245.7503357, 261.0979614, -451.4810486, 455.1615295
2: -277.0705872, 229.3539886, -357.8681030, 286.0145874, -563.0852051, 587.2220459
3: -114.6507111, 278.3254089, -144.4180450, 354.2912598, -468.9419556, 422.7434082
4: -309.5355530, 227.6612244, -398.9244995, 283.5452271, -593.0808105, 626.5856934

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4582488, upper bound: 808.8204571
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4474712, upper bound: 808.2818477
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2781429, upper bound: 808.9571169
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 41

Time for candidate selection: 7.21 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4571588, upper bound: 807.2531089
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4590576, upper bound: 808.9875678
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -350.1861877, 296.3976135, -320.1733704, 279.0426331, -629.2287598, 616.5709839
1: -280.1870117, 287.8131714, -255.9699402, 270.5928955, -550.7799072, 543.7830200
2: -406.6942444, 314.7042542, -372.7570801, 296.3858337, -703.0800781, 687.4613037
3: -159.5774536, 400.3084106, -149.8081055, 368.2790527, -527.8565063, 550.1165161
4: -453.2059937, 311.4501648, -415.4957581, 293.6712646, -746.8772583, 726.9456177

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8597900, upper bound: 808.8207767
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8560480, upper bound: 808.2819785
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4868850, upper bound: 808.9577728
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 41

Time for candidate selection: 7.14 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8586974, upper bound: 807.2532322
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8606171, upper bound: 808.9882473
time: 0.74 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 11.72 seconds
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.0716322, upper bound: 805.2289945
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.0717054, upper bound: 807.8292656
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.7506497, upper bound: 807.8675856
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.7506497, upper bound: 807.8244731
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.7676650, upper bound: 807.9257172
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.7676650, upper bound: 807.8958573
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.5669805, upper bound: 807.4482260
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.5669805, upper bound: 807.6034049
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.5669805, upper bound: 807.4482260
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.5669805, upper bound: 807.6034049
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.7642616, upper bound: 807.8191278
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.7643473, upper bound: 807.8191278
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.7559964, upper bound: 806.8390906
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.7561383, upper bound: 807.9115819
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.7835740, upper bound: 806.8391819
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -805.7837161, upper bound: 807.9325593
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 11.72
Output dim: 4, lower bound: -806.5976814, upper bound: 803.6419111
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 11.72
Output dim: 4, lower bound: -806.5976814, upper bound: 805.2933729
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -807.8663526, upper bound: 805.2778047
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -807.9359920, upper bound: 804.6427494
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 11.72
Output dim: 4, lower bound: -804.1947533, upper bound: 806.1418696
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 11.72
Output dim: 4, lower bound: -804.1939149, upper bound: 805.2898744
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 11.72
Output dim: 4, lower bound: -806.0014141, upper bound: 805.9488433
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 11.72
Output dim: 4, lower bound: -806.0014402, upper bound: 806.2166689
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 11.72
Output dim: 4, lower bound: -806.0014402, upper bound: 805.9488510
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 11.72
Output dim: 4, lower bound: -806.0014402, upper bound: 806.2172850
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -808.4571588, upper bound: 807.2531089
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -808.4590576, upper bound: 808.9875678
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -808.8586974, upper bound: 807.2532322
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 11.72
Output dim: 4, lower bound: -808.8606171, upper bound: 808.9882473
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.4589665, upper bound: 808.8504472
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.8605165, upper bound: 808.8512033
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.8209940, upper bound: 808.8200733
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.8209940, upper bound: 808.8212515
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.8212379, upper bound: 808.8365467
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.8212379, upper bound: 808.9886804
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.4581953, upper bound: 808.2840668
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.4581953, upper bound: 808.2842092
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.4585489, upper bound: 808.3258879
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.4585488, upper bound: 808.3260304
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.2830480, upper bound: 808.3322474
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.9766031, upper bound: 808.3327348
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.2830480, upper bound: 808.3322474
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.9766031, upper bound: 808.3327348
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.2324641, upper bound: 808.0497838
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.0667793, upper bound: 808.1704481
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.0667793, upper bound: 808.2530238
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.3415676, upper bound: 808.0667563
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.3415676, upper bound: 808.0667563
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.3416172, upper bound: 808.2532133
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.3416172, upper bound: 808.2532133
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -807.6063010, upper bound: 808.1402854
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.1888449, upper bound: 808.2002392
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -807.6062808, upper bound: 808.1402854
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.1888449, upper bound: 808.2002333
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.5462555, upper bound: 808.3421341
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.3415822, upper bound: 808.3415989
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.3800477, upper bound: 808.3290536
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.72
Output dim: 4, lower bound: -808.3289645, upper bound: 808.3285997
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=1011.34521484375
rel_dist={4: [-809.0065995903992, 809.0065995903992]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4685932, upper bound: 807.4327210
time: 0.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0035334, upper bound: 809.0035339
time: 0.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.62 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 4, lower bound: -806.4685932, upper bound: 807.4327210
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 4, lower bound: -809.0035334, upper bound: 809.0035339

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -355.7917175, 298.0696411, -435.6275024, 366.9198914, -722.7116089, 733.6970825
1: -284.4114380, 290.6557922, -349.4053650, 355.7280579, -640.1395264, 640.0610962
2: -411.3229675, 317.9904480, -507.5793762, 388.1117249, -799.4346924, 825.5696411
3: -162.9068451, 405.6134033, -197.9787903, 498.2431030, -661.1499634, 603.5921631
4: -458.9186401, 313.6753845, -565.4621582, 384.6995239, -843.6181030, 879.1375732

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4680720, upper bound: 806.4680720
time: 0.88 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4680720, upper bound: 807.4327210
time: 0.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -458.9994507, 383.7575378, -838.8145142, 839.6583252
1: -364.9006348, 369.0178833, -368.0681763, 372.0833130, -736.9838867, 737.0860596
2: -530.4962158, 402.6548462, -535.0563354, 405.9872131, -936.4833984, 937.7111816
3: -205.3341675, 519.1251221, -207.0673218, 523.5335693, -728.8677368, 726.1924438
4: -590.9782715, 398.9001770, -596.0374756, 402.1069641, -993.0852051, 994.9376221

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.4327210, upper bound: 806.4685932
time: 0.72 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.4327210, upper bound: 806.4685932
time: 0.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.38 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.38
Output dim: 4, lower bound: -806.4680720, upper bound: 806.4680720
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 4, lower bound: -806.4680720, upper bound: 807.4327210
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 4, lower bound: -807.4327210, upper bound: 806.4685932
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 4, lower bound: -807.4327210, upper bound: 806.4685932

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -355.7917175, 298.0696411, -453.8025818, 379.7243347, -735.5159912, 751.8721924
1: -284.4114380, 290.6557922, -363.8861084, 368.0965881, -652.5080566, 654.5418701
2: -411.3229675, 317.9904480, -529.0147095, 401.6497803, -812.9727783, 847.0051270
3: -162.9068451, 405.6134033, -204.8357544, 517.7385864, -680.6454468, 610.4489746
4: -458.9186401, 313.6753845, -589.3439331, 397.9489136, -856.8674927, 903.0192871

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4195833, upper bound: 807.2151597
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4680337, upper bound: 807.4303342
time: 0.77 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -355.7917175, 298.0696411, -753.1265869, 736.4506836
1: -364.9006348, 369.0178833, -284.4114380, 290.6557922, -655.5563965, 653.4293213
2: -530.4962158, 402.6548462, -411.3229675, 317.9904480, -848.4865723, 813.9777832
3: -205.3341675, 519.1251221, -162.9068451, 405.6134033, -610.9473877, 682.0319824
4: -590.9782715, 398.9001770, -458.9186401, 313.6753845, -904.6536865, 857.8186646

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.4326952, upper bound: 806.4685230
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.4303305, upper bound: 806.4685282
time: 0.68 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -455.0569763, 380.6590271, -835.7160034, 835.7160034
1: -364.9006348, 369.0178833, -364.9006348, 369.0178833, -733.9185181, 733.9185181
2: -530.4962158, 402.6548462, -530.4962158, 402.6548462, -933.1510620, 933.1510620
3: -205.3341675, 519.1251221, -205.3341675, 519.1251221, -724.4591675, 724.4591675
4: -590.9782715, 398.9001770, -590.9782715, 398.9001770, -989.8784180, 989.8782959

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.4326993, upper bound: 808.5813317
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.4303342, upper bound: 808.5812559
time: 0.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.40 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.40
Output dim: 4, lower bound: -806.4195833, upper bound: 807.2151597
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -806.4680337, upper bound: 807.4303342
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -807.4326952, upper bound: 806.4685230
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -807.4303305, upper bound: 806.4685282
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -807.4326993, upper bound: 808.5813317
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 4, lower bound: -807.4303342, upper bound: 808.5812559

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -346.6198730, 290.6638184, -448.0665588, 375.0095520, -721.6293335, 738.7302856
1: -276.9523926, 283.5098572, -359.2083130, 363.5391541, -640.4914551, 642.7181396
2: -400.2306824, 310.2190247, -522.0534058, 396.6775208, -796.9082031, 832.2724609
3: -158.6000671, 395.0196533, -202.3629150, 511.1264648, -669.7265015, 597.3825073
4: -446.5895996, 306.0231628, -581.6328125, 393.0583191, -839.6479492, 887.6560059

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4121961, upper bound: 807.1469671
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4684710, upper bound: 807.4303305
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -351.6240845, 294.5337219, -732.3060913, 719.0653687
1: -350.6974182, 356.2439270, -281.0322876, 287.1985779, -637.8959961, 637.2762451
2: -509.7792053, 388.8671265, -406.3869934, 314.1864014, -823.9655151, 795.2541504
3: -198.0632477, 499.4380493, -161.1513062, 400.8664856, -598.9297485, 660.5892334
4: -568.3402710, 385.4368286, -453.4685974, 309.9019165, -878.2421875, 838.9053345

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2893259, upper bound: 805.2571493
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.2571775, upper bound: 806.2177793
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -350.8417969, 294.0659485, -738.9554443, 723.1326904
1: -356.6080017, 360.9292603, -280.3814087, 286.7947693, -643.4027710, 641.3106689
2: -518.1353760, 393.8309326, -405.3389282, 313.7921753, -831.9275513, 799.1698608
3: -200.9437408, 507.3840332, -160.5826263, 399.8922424, -600.8359985, 667.9666138
4: -577.2812500, 390.2192383, -452.2669067, 309.5393066, -886.8205566, 842.4859009

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7205241, upper bound: 805.8241682
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8669888, upper bound: 805.8372240
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1996605, upper bound: 806.2175898
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -449.0044861, 375.2546387, -813.0269165, 816.4458008
1: -350.6974182, 356.2439270, -359.9841003, 363.8726196, -714.5700684, 716.2280273
2: -509.7792053, 388.8671265, -523.2807617, 397.1494751, -906.9286499, 912.1478882
3: -198.0632477, 499.4380493, -202.6083221, 512.1017456, -710.1649780, 702.0463257
4: -568.3402710, 385.4368286, -582.9922485, 393.3094788, -961.6496582, 968.4290771

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814071, upper bound: 808.5811934
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5814071, upper bound: 808.5811934
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -449.1309204, 375.7961121, -820.6854248, 821.4218140
1: -356.6080017, 360.9292603, -360.0684814, 364.3156738, -720.9236450, 720.9977417
2: -518.1353760, 393.8309326, -523.3109131, 397.5250549, -915.6604004, 917.1418457
3: -200.9437408, 507.3840332, -202.7751923, 512.2996216, -713.2432861, 710.1591797
4: -577.2812500, 390.2192383, -583.0198364, 393.8540344, -971.1352539, 973.2390137

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5168603, upper bound: 808.2533364
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5813245, upper bound: 808.5812322
time: 0.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.76 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 4, lower bound: -806.4121961, upper bound: 807.1469671
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 4, lower bound: -806.4684710, upper bound: 807.4303305
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 4, lower bound: -806.2893259, upper bound: 805.2571493
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 4, lower bound: -807.2571775, upper bound: 806.2177793
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 4, lower bound: -806.8669888, upper bound: 805.8372240
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 4, lower bound: -807.1996605, upper bound: 806.2175898
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 4, lower bound: -808.5814071, upper bound: 808.5811934
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 4, lower bound: -808.5814071, upper bound: 808.5811934
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 4, lower bound: -808.5168603, upper bound: 808.2533364
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 4, lower bound: -808.5813245, upper bound: 808.5812322

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -340.2148132, 285.5393066, -440.6028442, 368.4757996, -708.6906128, 726.1421509
1: -271.7571716, 278.5577393, -353.1407166, 357.2431946, -629.0003662, 631.6984863
2: -392.7020569, 304.8424683, -513.2144165, 389.9300232, -782.6320801, 818.0568237
3: -155.9787292, 387.8564453, -198.9626770, 502.4864197, -658.4651489, 586.8190918
4: -438.3005676, 300.7043762, -571.8695068, 386.2991943, -824.5997314, 872.5738525

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8371412, upper bound: 806.8669917
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2175252, upper bound: 807.1996636
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -437.7723999, 367.4412842, -805.2136230, 805.2136230
1: -350.6974182, 356.2439270, -350.6974182, 356.2439270, -706.9413452, 706.9413452
2: -509.7792053, 388.8671265, -509.7792053, 388.8671265, -898.6461792, 898.6463013
3: -198.0632477, 499.4380493, -198.0632477, 499.4380493, -697.5012817, 697.5012817
4: -568.3402710, 385.4368286, -568.3402710, 385.4368286, -953.7770996, 953.7770996

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5374245, upper bound: 808.3289453
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7918140, upper bound: 808.3293974
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -444.8894653, 372.2908936, -810.0632935, 812.3307495
1: -350.6974182, 356.2439270, -356.6080017, 360.9292603, -711.6267090, 712.8518677
2: -509.7792053, 388.8671265, -518.1353760, 393.8309326, -903.6101074, 907.0024414
3: -198.0632477, 499.4380493, -200.9437408, 507.3840332, -705.4472656, 700.3817139
4: -568.3402710, 385.4368286, -577.2812500, 390.2192383, -958.5594482, 962.7180786

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8132542, upper bound: 807.9772166
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9965816, upper bound: 808.5813312
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -424.3577271, 356.3072205, -404.9446411, 341.5538330, -765.9114990, 761.2518311
1: -340.0241394, 345.3395996, -324.4280701, 330.8679504, -670.8920898, 669.7674561
2: -494.0138855, 376.8282471, -471.4970703, 361.0960388, -855.1098633, 848.3253174
3: -192.1097107, 483.9433899, -183.9692078, 461.8945618, -654.0042725, 667.9125977
4: -550.6756592, 373.5756226, -525.8263550, 358.2057800, -908.8814697, 899.4019775

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5152698, upper bound: 808.2532224
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2530670, upper bound: 808.2529717
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2530669, upper bound: 808.2529717
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -417.9097900, 349.0167542, -686.8149414, 556.9207764, -971.1631470, 1035.8316650
1: -334.6179199, 338.4356995, -552.1990967, 540.7856445, -871.9020996, 890.6347656
2: -485.6710510, 369.7235718, -798.9375000, 588.8789062, -1070.9394531, 1168.6611328
3: -189.1661987, 475.4371948, -303.5726318, 772.9661865, -962.1323242, 775.6733398
4: -541.3536987, 366.2839050, -888.2079468, 582.1496582, -1120.1715088, 1254.4918213

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532388, upper bound: 808.2638665
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2532388, upper bound: 808.5812322
time: 0.71 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.02 seconds
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 4, lower bound: -805.8371412, upper bound: 806.8669917
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.02
Output dim: 4, lower bound: -806.2175252, upper bound: 807.1996636
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 4, lower bound: -808.5374245, upper bound: 808.3289453
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 4, lower bound: -808.7918140, upper bound: 808.3293974
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 4, lower bound: -807.8132542, upper bound: 807.9772166
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 4, lower bound: -808.9965816, upper bound: 808.5813312
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 4, lower bound: -808.2530670, upper bound: 808.2529717
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 4, lower bound: -808.2530669, upper bound: 808.2529717
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 4, lower bound: -808.2532388, upper bound: 808.2638665
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.02
Output dim: 4, lower bound: -808.2532388, upper bound: 808.5812322

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -300.6821594, 266.6185913, -373.7425537, 320.4878540, -621.1699829, 640.3610840
1: -240.2742615, 258.2109070, -298.9227600, 310.6056213, -550.8798828, 557.1336670
2: -350.0690002, 282.6289978, -434.7728271, 339.6759033, -689.7448730, 717.4016724
3: -142.5813751, 348.4453430, -172.0658264, 428.4988098, -571.0802002, 520.5111084
4: -390.8204651, 280.6113586, -484.9254150, 336.8037109, -727.6240234, 765.5367432

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5373903, upper bound: 808.5370416
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5373903, upper bound: 809.0014449
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -424.1921692, 356.3082275, -429.9951477, 361.0730591, -785.2650146, 786.3033447
1: -339.7845459, 345.6167603, -344.4439087, 350.1649475, -689.9494629, 690.0606079
2: -493.6854553, 377.4214478, -500.5626221, 382.3164673, -876.0019531, 877.9840698
3: -192.2233276, 484.1344299, -194.7217407, 490.6405640, -682.8638916, 678.8562012
4: -550.4783936, 373.9692078, -558.1074219, 378.8787842, -929.3571167, 932.0766602

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7880737, upper bound: 808.2829429
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0026653, upper bound: 809.0024242
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -402.8525085, 341.1595459, -424.0348816, 356.7133789, -759.5659180, 765.1942139
1: -322.5713196, 330.5337524, -339.8127747, 345.6748352, -668.2459717, 670.3464355
2: -469.0149536, 360.7241211, -493.8036804, 377.1585693, -846.1735229, 854.5277710
3: -183.2427368, 460.5935669, -192.2160950, 484.2398071, -667.4825439, 652.8095703
4: -523.3585815, 358.0090332, -550.4446411, 374.0341797, -897.3927612, 908.4536743

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5332251, upper bound: 807.6077421
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8128323, upper bound: 807.9772259
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -505.0977478, 420.0357666, -438.7721252, 367.5714111, -872.6690674, 858.8078613
1: -405.1225281, 407.2164612, -351.6248779, 356.2838135, -761.4063721, 758.8413086
2: -588.5635986, 444.3673096, -510.8882141, 388.7623291, -977.3258667, 955.2554932
3: -227.9976196, 574.9717407, -198.3679657, 500.4271851, -728.4247437, 773.3397217
4: -655.8900146, 440.8576050, -569.3222656, 385.3018494, -1041.1917725, 1010.1798096

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9958522, upper bound: 808.3807762
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7918140, upper bound: 808.3293803
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -400.8190918, 338.1311340, -404.9446411, 341.5538330, -742.3729248, 743.0756836
1: -321.0563354, 327.5602722, -324.4280701, 330.8679504, -651.9241943, 651.9881592
2: -466.4544678, 357.4834900, -471.4970703, 361.0960388, -827.5505371, 828.9805908
3: -182.1852417, 457.0790405, -183.9692078, 461.8945618, -644.0797119, 641.0482178
4: -520.2389526, 354.6555786, -525.8263550, 358.2057800, -878.4447021, 880.4817505

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.6209876, upper bound: 806.9955693
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.6204839, upper bound: 806.6203269
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -679.6831055, 551.4983521, -404.9446411, 341.5538330, -1021.2369385, 952.3242798
1: -546.4172363, 535.5155029, -324.4280701, 330.8679504, -877.2851562, 856.1009521
2: -790.4291382, 583.1304932, -471.4970703, 361.0960388, -1151.5251465, 1050.4163818
3: -300.6394653, 764.9594727, -183.9692078, 461.8945618, -758.6663818, 948.9287109
4: -878.8193970, 576.5201416, -525.8263550, 358.2057800, -1237.0250244, 1098.3654785

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.6209876, upper bound: 807.3056998
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.6204841, upper bound: 806.6203269
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -398.6757812, 336.2539673, -682.0446777, 553.3347778, -948.3328857, 1018.2986450
1: -319.2977600, 325.5302124, -548.3287964, 537.3009644, -853.1141968, 873.8588257
2: -463.9438477, 355.1840210, -793.3260498, 585.1173706, -1045.2647705, 1148.5100098
3: -181.0929260, 454.5400085, -301.6544800, 767.8012085, -948.8941040, 752.6765137
4: -517.4816895, 352.5979614, -882.0245972, 578.4559937, -1092.3950195, 1234.6224365

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2530198, upper bound: 808.2638665
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2530198, upper bound: 808.2529717
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -683.3704834, 554.1118774, -687.3020020, 557.3321533, -1231.1501465, 1232.1099854
1: -549.4003906, 538.0770874, -552.5928955, 541.1862793, -1082.9442139, 1083.2122803
2: -794.7464600, 585.9254150, -799.5101929, 589.3105469, -1375.2606201, 1376.8990479
3: -302.0794983, 768.9109497, -303.7902832, 773.5118408, -1070.3320312, 1067.3435059
4: -883.5769653, 579.2427979, -888.8448486, 582.5694580, -1456.5795898, 1458.8233643

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.6207938, upper bound: 807.4003274
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5452730, upper bound: 807.7259669
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.5452772, upper bound: 805.5452719
time: 0.75 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.70 seconds
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 4, lower bound: -808.5373903, upper bound: 808.5370416
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 4, lower bound: -808.5373903, upper bound: 809.0014449
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 4, lower bound: -808.7880737, upper bound: 808.2829429
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 4, lower bound: -809.0026653, upper bound: 809.0024242
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 4, lower bound: -807.5332251, upper bound: 807.6077421
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 4, lower bound: -807.8128323, upper bound: 807.9772259
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 4, lower bound: -808.9958522, upper bound: 808.3807762
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 4, lower bound: -808.7918140, upper bound: 808.3293803
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.70
Output dim: 4, lower bound: -806.6209876, upper bound: 806.9955693
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.70
Output dim: 4, lower bound: -806.6204839, upper bound: 806.6203269
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.70
Output dim: 4, lower bound: -806.6209876, upper bound: 807.3056998
IS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.70
Output dim: 4, lower bound: -806.6204841, upper bound: 806.6203269
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 4, lower bound: -808.2530198, upper bound: 808.2638665
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 4, lower bound: -808.2530198, upper bound: 808.2529717
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 4, lower bound: -805.5452730, upper bound: 807.7259669
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.70
Output dim: 4, lower bound: -805.5452772, upper bound: 805.5452719

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -300.6821594, 266.6185913, -300.6821594, 266.6185913, -567.3006592, 567.3007202
1: -240.2742615, 258.2109070, -240.2742615, 258.2109070, -498.4851685, 498.4851685
2: -350.0690002, 282.6289978, -350.0690002, 282.6289978, -632.6979980, 632.6979980
3: -142.5813751, 348.4453430, -142.5813751, 348.4453430, -491.0267029, 491.0267029
4: -390.8204651, 280.6113586, -390.8204651, 280.6113586, -671.4318237, 671.4318237

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5105147, upper bound: 806.4892600
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4899148, upper bound: 806.4892675
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -300.6821594, 266.6185913, -424.1275635, 356.1332092, -656.8153687, 690.7460938
1: -240.2742615, 258.2109070, -339.7343445, 345.4415283, -585.7158203, 597.9452515
2: -350.0690002, 282.6289978, -493.6127319, 377.2241211, -727.2930908, 776.2416382
3: -142.5813751, 348.4453430, -192.1471100, 484.0213928, -626.6025391, 540.5923462
4: -390.8204651, 280.6113586, -550.3928833, 373.7930603, -764.6134644, 831.0042114

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5105147, upper bound: 806.7089646
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4899148, upper bound: 806.7089718
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -403.6827393, 341.1568298, -394.8012390, 334.8991089, -738.5817871, 735.9578857
1: -323.2937317, 330.8091125, -316.1155090, 324.5741577, -647.8679199, 646.9246216
2: -469.7868347, 361.2003784, -459.5066223, 354.3055420, -824.0924072, 820.7069702
3: -183.6711273, 461.4839172, -179.9227905, 451.6310425, -635.3021240, 641.4067383
4: -524.0942383, 358.1392822, -512.7974243, 351.5378113, -875.6319580, 870.9367065

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 20

Time for candidate selection: 4.17 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6318664, upper bound: 808.0068318
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7351418, upper bound: 808.2367717
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -417.7650146, 351.3507996, -498.0815125, 414.1718140, -831.9368286, 849.4322510
1: -334.5597534, 340.7451172, -399.5077515, 401.6736450, -736.2333984, 740.2528687
2: -486.0765686, 372.1083679, -580.2825928, 438.3619995, -924.4385986, 952.3909302
3: -189.5283356, 476.8426514, -225.0961761, 567.1225586, -756.6508789, 701.9388428
4: -542.1068115, 368.8151855, -646.6953125, 434.8447571, -976.9515381, 1015.5104980

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9747183, upper bound: 806.7098820
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7100904, upper bound: 806.7098905
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -339.5380249, 295.6781616, -293.7804871, 261.7765198, -601.3145142, 589.4585571
1: -271.3579712, 286.3671875, -234.7945251, 253.3287048, -524.6866455, 521.1617432
2: -394.6839905, 313.0200500, -341.8663025, 277.0762939, -671.7601929, 654.8863525
3: -158.2649384, 390.7572937, -139.7844696, 340.9545898, -499.2195129, 530.5417480
4: -440.9541931, 310.8043518, -381.7980042, 275.2938232, -716.2480469, 692.6022339

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 30

Time for candidate selection: 4.13 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.5197591, upper bound: 804.0821051
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.0552544, upper bound: 807.2360094
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3795427, upper bound: 807.2369872
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -394.8012390, 334.8991089, -411.1617432, 346.1337585, -740.9349976, 746.0607300
1: -316.1155090, 324.5741577, -329.4780579, 335.5887451, -651.7042236, 654.0522461
2: -459.5066223, 354.3055420, -478.5378113, 366.2790527, -825.7856445, 832.8433838
3: -179.9227905, 451.6310425, -186.6743469, 469.8007202, -649.7235107, 638.3052979
4: -512.7974243, 351.5378113, -533.4836426, 363.0794983, -875.8768921, 885.0213623

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 41

Time for candidate selection: 4.12 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8128284, upper bound: 807.8744794
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.2883910, upper bound: 807.7222408
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7368068, upper bound: 807.7549332
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -438.2236328, 371.2074890, -306.6577454, 270.5864868, -708.8101196, 677.8651733
1: -351.0169983, 359.7406616, -245.1115265, 261.8984985, -612.9155273, 604.8521729
2: -510.0557861, 392.8886108, -356.8811340, 286.5234985, -796.5792847, 749.7696533
3: -200.4833069, 500.6698303, -144.7649994, 354.8236389, -555.3069458, 645.4348145
4: -568.9026489, 389.9518127, -398.2102661, 284.4723816, -853.3748779, 788.1621094

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5374063, upper bound: 808.3289453
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5374063, upper bound: 808.3293803
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -498.0815125, 414.1718140, -425.3565063, 356.4174805, -854.4990234, 839.5283203
1: -399.5077515, 401.6736450, -340.8368835, 345.6394958, -745.1472168, 742.5104980
2: -580.2825928, 438.3619995, -494.9598389, 377.4329834, -957.7155762, 933.3217773
3: -225.0961761, 567.1225586, -192.5426636, 485.2558594, -710.3518677, 759.6651001
4: -646.6953125, 434.8447571, -551.6343384, 373.8498230, -1020.5451660, 986.4790649

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5374245, upper bound: 808.3289453
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5374245, upper bound: 808.3293803
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -398.6757812, 336.2539673, -668.1148071, 542.8779297, -937.5998535, 1004.3686523
1: -319.2977600, 325.5302124, -537.0526733, 527.3202515, -842.9237671, 862.5828857
2: -463.9438477, 355.1840210, -777.0159302, 574.2546387, -1034.1740723, 1132.1999512
3: -181.0929260, 454.5400085, -295.9263611, 752.4465332, -933.5394287, 746.7434692
4: -517.4816895, 352.5979614, -864.1340942, 567.7005615, -1081.3842773, 1216.7319336

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36

Time for candidate selection: 4.42 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6058509, upper bound: 807.9613000
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1886834, upper bound: 808.2000722
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -398.6757812, 336.2539673, -678.2709351, 550.2514648, -945.3000488, 1014.5249023
1: -319.2977600, 325.5302124, -545.2637939, 534.3253784, -850.1760864, 870.7940063
2: -463.9438477, 355.1840210, -788.7483521, 581.8750000, -1042.0699463, 1143.9323730
3: -181.0929260, 454.5400085, -300.0126953, 763.3845825, -944.4774780, 751.0534058
4: -517.4816895, 352.5979614, -876.9635620, 575.2696533, -1089.2652588, 1229.5614014

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36

Time for candidate selection: 4.41 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6058307, upper bound: 807.9612837
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1886834, upper bound: 808.2000722
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -683.3704834, 554.1118774, -685.7027588, 556.0025635, -1229.7562256, 1230.3692627
1: -549.4003906, 538.0770874, -551.2969971, 539.8915405, -1081.5983887, 1081.8056641
2: -794.7464600, 585.9254150, -797.6085815, 587.9030762, -1373.7911377, 1374.8446045
3: -302.0794983, 768.9109497, -303.0865784, 771.6430664, -1068.4359131, 1066.6011963
4: -883.5769653, 579.2427979, -886.7474365, 581.1903076, -1455.1329346, 1456.5869141

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8605921, upper bound: 806.8604793
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8605921, upper bound: 806.8605966
time: 0.99 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.83 seconds
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -808.5105147, upper bound: 806.4892600
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 4, lower bound: -806.4899148, upper bound: 806.4892675
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -808.5105147, upper bound: 806.7089646
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 4, lower bound: -806.4899148, upper bound: 806.7089718
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -807.6318664, upper bound: 808.0068318
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -808.7351418, upper bound: 808.2367717
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -808.9747183, upper bound: 806.7098820
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 4, lower bound: -806.7100904, upper bound: 806.7098905
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 4, lower bound: -807.0552544, upper bound: 807.2360094
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 4, lower bound: -807.3795427, upper bound: 807.2369872
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -807.2883910, upper bound: 807.7222408
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -807.7368068, upper bound: 807.7549332
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -808.5374063, upper bound: 808.3289453
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -808.5374063, upper bound: 808.3293803
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -808.5374245, upper bound: 808.3289453
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -808.5374245, upper bound: 808.3293803
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -807.6058509, upper bound: 807.9613000
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -808.1886834, upper bound: 808.2000722
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -807.6058307, upper bound: 807.9612837
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.83
Output dim: 4, lower bound: -808.1886834, upper bound: 808.2000722
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 4, lower bound: -806.8605921, upper bound: 806.8604793
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.83
Output dim: 4, lower bound: -806.8605921, upper bound: 806.8605966

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -298.7503052, 265.2289429, -300.6821594, 266.6185913, -565.3688965, 565.9110718
1: -238.6868286, 256.8603516, -240.2742615, 258.2109070, -496.8976135, 497.1346130
2: -347.7602234, 281.1653442, -350.0690002, 282.6289978, -630.3892212, 631.2343750
3: -141.8453217, 346.3019714, -142.5813751, 348.4453430, -490.2906494, 488.8833618
4: -388.2952271, 279.1665039, -390.8204651, 280.6113586, -668.9066162, 669.9869385

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4899128, upper bound: 806.4892600
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4899128, upper bound: 806.4892600
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -298.7503052, 265.2289429, -424.1275635, 356.1332092, -654.8835449, 689.3565063
1: -238.6868286, 256.8603516, -339.7343445, 345.4415283, -584.1282349, 596.5946045
2: -347.7602234, 281.1653442, -493.6127319, 377.2241211, -724.9843750, 774.7780762
3: -141.8453217, 346.3019714, -192.1471100, 484.0213928, -625.8666382, 538.4490967
4: -388.2952271, 279.1665039, -550.3928833, 373.7930603, -762.0881958, 829.5592651

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4903412, upper bound: 806.7089619
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4903412, upper bound: 806.7089619
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -384.8565674, 327.4353027, -394.8012390, 334.8991089, -719.7555542, 722.2363892
1: -308.0140686, 317.4070129, -316.1155090, 324.5741577, -632.5882568, 633.5225220
2: -447.3511047, 346.6392212, -459.5066223, 354.3055420, -801.6566162, 806.1458740
3: -175.8584137, 440.2359009, -179.9227905, 451.6310425, -627.4893188, 620.1586914
4: -499.5995483, 343.7287292, -512.7974243, 351.5378113, -851.1372070, 856.5261230

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5033930, upper bound: 805.0949137
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6314504, upper bound: 808.0062200
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -417.8883972, 362.6159973, -394.8012390, 334.8991089, -752.7874756, 757.4171753
1: -334.9901428, 351.1107178, -316.1155090, 324.5741577, -659.5643311, 667.2261963
2: -487.2348328, 383.1231384, -459.5066223, 354.3055420, -841.5404053, 842.6297607
3: -194.2797546, 480.6300354, -179.9227905, 451.6310425, -645.9107056, 660.5528564
4: -543.6226807, 380.1124268, -512.7974243, 351.5378113, -895.1604614, 892.9097900

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0992995, upper bound: 805.1477360
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7346345, upper bound: 808.2361530
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -416.0515442, 350.0075073, -498.0815125, 414.1718140, -830.2233887, 848.0889893
1: -333.1695862, 339.4431458, -399.5077515, 401.6736450, -734.8432007, 738.9509277
2: -484.0410461, 370.6898193, -580.2825928, 438.3619995, -922.4030151, 950.9723511
3: -188.8265839, 474.9241638, -225.0961761, 567.1225586, -755.9491577, 700.0202637
4: -539.8623047, 367.4197693, -646.6953125, 434.8447571, -974.7070312, 1014.1149292

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9741304, upper bound: 806.4902137
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9741304, upper bound: 806.7098820
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -378.3058472, 322.7847900, -411.1617432, 346.1337585, -724.4395752, 733.9464111
1: -302.7047424, 312.7461548, -329.4780579, 335.5887451, -638.2933960, 642.2242432
2: -439.8032227, 341.5080872, -478.5378113, 366.2790527, -806.0822754, 820.0458984
3: -173.1756287, 432.9067993, -186.6743469, 469.8007202, -642.9763184, 619.5810547
4: -491.3053589, 338.8619080, -533.4836426, 363.0794983, -854.3848877, 872.3455811

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.0637460, upper bound: 806.6353509
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.0637460, upper bound: 807.7222408
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -410.7887878, 358.0437622, -411.1617432, 346.1337585, -756.9225464, 769.2055054
1: -329.2294006, 346.5978088, -329.4780579, 335.5887451, -664.8180542, 676.0758667
2: -479.0015869, 378.0619507, -478.5378113, 366.2790527, -845.2805176, 856.5997314
3: -191.5332642, 472.7409668, -186.6743469, 469.8007202, -661.3339844, 659.4152222
4: -534.5559082, 375.0802307, -533.4836426, 363.0794983, -897.6353760, 908.5637207

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3558008, upper bound: 806.6359720
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.3558008, upper bound: 807.7549447
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -362.4513855, 315.4191895, -306.6577454, 270.5864868, -633.0377808, 622.0769043
1: -290.1376953, 305.3040161, -245.1115265, 261.8984985, -552.0361938, 550.4154053
2: -422.3479309, 333.7776184, -356.8811340, 286.5234985, -708.8713379, 690.6587524
3: -169.6179810, 417.7650757, -144.7649994, 354.8236389, -524.4415283, 562.5300903
4: -471.5178528, 331.7652893, -398.2102661, 284.4723816, -755.9901123, 729.9755859

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 33

Time for candidate selection: 4.36 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5059337, upper bound: 807.6778994
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5374063, upper bound: 808.3800330
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -492.6842041, 409.8632812, -306.6577454, 270.5864868, -763.2706909, 716.5209351
1: -395.1902466, 397.5688782, -245.1115265, 261.8984985, -657.0887451, 642.6802368
2: -573.9177246, 433.9119568, -356.8811340, 286.5234985, -860.4411621, 790.7930908
3: -222.9507751, 561.1869507, -144.7649994, 354.8236389, -577.7743530, 705.9519653
4: -639.6378174, 430.3858337, -398.2102661, 284.4723816, -924.1101074, 828.5960693

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 33

Time for candidate selection: 4.46 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5059342, upper bound: 807.6781657
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5374063, upper bound: 808.3807762
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -362.4513855, 315.4191895, -425.3565063, 356.4174805, -718.8688965, 740.7756958
1: -290.1376953, 305.3040161, -340.8368835, 345.6394958, -635.7772217, 646.1408691
2: -422.3479309, 333.7776184, -494.9598389, 377.4329834, -799.7808228, 828.7374268
3: -169.6179810, 417.7650757, -192.5426636, 485.2558594, -654.8737183, 610.3075562
4: -471.5178528, 331.7652893, -551.6343384, 373.8498230, -845.3676147, 883.3995972

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 43

Time for candidate selection: 4.50 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4888551, upper bound: 807.5025388
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5374063, upper bound: 808.3023125
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -492.6842041, 409.8632812, -425.3565063, 356.4174805, -849.1016846, 835.2197266
1: -395.1902466, 397.5688782, -340.8368835, 345.6394958, -740.8297119, 738.4057617
2: -573.9177246, 433.9119568, -494.9598389, 377.4329834, -951.3506470, 928.8718262
3: -222.9507751, 561.1869507, -192.5426636, 485.2558594, -708.2065430, 753.7294922
4: -639.6378174, 430.3858337, -551.6343384, 373.8498230, -1013.4876709, 982.0201416

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 43

Time for candidate selection: 4.48 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4888551, upper bound: 807.5027412
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5374063, upper bound: 808.3027474
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -372.7213440, 317.6488037, -668.1148071, 542.8779297, -911.4733276, 985.7636108
1: -298.1468811, 307.3338623, -537.0526733, 527.3202515, -821.6243286, 844.3864746
2: -432.8846130, 335.5460510, -777.0159302, 574.2546387, -1002.8058472, 1112.5620117
3: -170.3899536, 425.0518188, -295.9263611, 752.4465332, -922.8364868, 717.1201782
4: -483.5900574, 333.3209229, -864.1340942, 567.7005615, -1047.1627197, 1197.4548340

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 11

Time for candidate selection: 4.65 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6052280, upper bound: 807.6218848
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6052280, upper bound: 807.9613001
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -423.2725830, 372.8086548, -668.1148071, 542.8779297, -961.4952393, 1040.9233398
1: -339.4468994, 361.0475159, -537.0526733, 527.3202515, -862.4977417, 898.1002197
2: -493.0749207, 393.8171082, -777.0159302, 574.2546387, -1062.4326172, 1170.8330078
3: -199.5374146, 486.4467468, -295.9263611, 752.4465332, -951.9839478, 778.4558716
4: -549.4050903, 389.0480957, -864.1340942, 567.7005615, -1112.2760010, 1253.1820068

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 11

Time for candidate selection: 4.67 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1886834, upper bound: 808.2051492
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1886834, upper bound: 808.2050352
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1885849, upper bound: 808.2047876
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -372.7213440, 317.6488037, -678.2709351, 550.2514648, -919.1734619, 995.9197388
1: -298.1468811, 307.3338623, -545.2637939, 534.3253784, -828.8765869, 852.5975342
2: -432.8846130, 335.5460510, -788.7483521, 581.8750000, -1010.7015381, 1124.2944336
3: -170.3899536, 425.0518188, -300.0126953, 763.3845825, -933.7745361, 721.4300537
4: -483.5900574, 333.3209229, -876.9635620, 575.2696533, -1055.0437012, 1210.2843018

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5062182, upper bound: 807.7802355
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=1011.34521484375
rel_dist={4: [-809.0063734281499, 809.00637342815]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1131.00 seconds
