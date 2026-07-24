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
execution time: IAR + LP analysis = 1.57 + 1.93 = 3.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -809.0067386, upper bound: 809.0067386


# Binary Search by BASE starts (time budget: 1196.50 seconds, max iter: 100)

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
Binary search time: 66.63 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1129.87 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.4692061, upper bound: 809.0031754
time: 0.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0039002, upper bound: 809.0039007
time: 0.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.51 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 4, lower bound: -806.4692061, upper bound: 809.0031754
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 4, lower bound: -809.0039002, upper bound: 809.0039007

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -355.7917175, 298.0696411, -465.1231079, 388.6268616, -744.4185181, 763.1926880
1: -284.4114380, 290.6557922, -372.9984131, 376.8692322, -661.2806396, 663.6541748
2: -411.3229675, 317.9904480, -542.1752319, 411.2849731, -822.6077881, 860.1655884
3: -162.9068451, 405.6134033, -209.7512207, 530.4779663, -693.3848267, 615.3645630
4: -458.9186401, 313.6753845, -603.9422607, 407.1642456, -866.0828857, 917.6176758

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

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
time: 0.68 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -465.2375793, 388.7125854, -843.7695312, 845.8966064
1: -364.9006348, 369.0178833, -373.0900269, 376.9524841, -741.8531494, 742.1078491
2: -530.4962158, 402.6548462, -542.3095093, 411.3765564, -941.8726807, 944.9643555
3: -205.3341675, 519.1251221, -209.7973175, 530.6040649, -735.9381104, 728.9224243
4: -590.9782715, 398.9001770, -604.0916138, 407.2537537, -998.2319336, 1002.9916992

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

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
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.23 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.23
Output dim: 4, lower bound: -806.4684808, upper bound: 806.4684808
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 3.23
Output dim: 4, lower bound: -806.4684808, upper bound: 806.4684808
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.23
Output dim: 4, lower bound: -809.0031754, upper bound: 806.4692061
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.23
Output dim: 4, lower bound: -809.0031754, upper bound: 806.4692061

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -355.7917175, 298.0696411, -753.1265869, 736.4506836
1: -364.9006348, 369.0178833, -284.4114380, 290.6557922, -655.5563965, 653.4293213
2: -530.4962158, 402.6548462, -411.3229675, 317.9904480, -848.4865723, 813.9777832
3: -205.3341675, 519.1251221, -162.9068451, 405.6134033, -610.9473877, 682.0319824
4: -590.9782715, 398.9001770, -458.9186401, 313.6753845, -904.6536865, 857.8186646

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0025439, upper bound: 806.4690137
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7687614, upper bound: 805.8383925
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9954292, upper bound: 806.2185391
time: 0.68 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -455.0569763, 380.6590271, -835.7160034, 835.7160034
1: -364.9006348, 369.0178833, -364.9006348, 369.0178833, -733.9185181, 733.9185181
2: -530.4962158, 402.6548462, -530.4962158, 402.6548462, -933.1510620, 933.1510620
3: -205.3341675, 519.1251221, -205.3341675, 519.1251221, -724.4591675, 724.4591675
4: -590.9782715, 398.9001770, -590.9782715, 398.9001770, -989.8784180, 989.8782959

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0025444, upper bound: 808.8222452
time: 0.75 seconds

## Relational analysis of IS_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0025031, upper bound: 808.5813317
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5808102, upper bound: 806.4690029
time: 1.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.27 seconds
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.27
Output dim: 4, lower bound: -808.7687614, upper bound: 805.8383925
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.27
Output dim: 4, lower bound: -808.9954292, upper bound: 806.2185391
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.27
Output dim: 4, lower bound: -809.0025031, upper bound: 808.5813317
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.27
Output dim: 4, lower bound: -808.5808102, upper bound: 806.4690029

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -451.4883728, 378.0201416, -251.5948334, 219.6874542, -671.1757812, 629.6149902
1: -362.0157776, 366.4407043, -200.1122742, 214.7951965, -576.8109741, 566.5529175
2: -526.2915039, 399.8591919, -288.3356018, 235.8116150, -762.1031494, 688.1945801
3: -203.8824615, 515.1088867, -120.3001709, 288.3111572, -492.1936035, 635.4090576
4: -586.3282471, 396.1748047, -322.8778687, 232.7944031, -819.1225586, 719.0526733

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8732163, upper bound: 805.8383042
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7105309, upper bound: 804.9129156
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5187486, upper bound: 805.8381264
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -348.0386353, 292.4441223, -747.5010986, 728.6975708
1: -364.9006348, 369.0178833, -278.1151123, 285.1520081, -650.0526123, 647.1329956
2: -530.4962158, 402.6548462, -402.1842957, 312.0704346, -842.5665894, 804.8391113
3: -205.3341675, 519.1251221, -159.7416534, 396.7953796, -602.1293945, 678.8667603
4: -590.9782715, 398.9001770, -448.7807617, 307.8369446, -898.8151855, 847.6809082

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9949002, upper bound: 806.2182653
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2182480
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5189495, upper bound: 806.2182480
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -455.0569763, 380.6590271, -818.4312744, 822.4982910
1: -350.6974182, 356.2439270, -364.9006348, 369.0178833, -719.7153320, 721.1445312
2: -509.7792053, 388.8671265, -530.4962158, 402.6548462, -912.4340820, 919.3633423
3: -198.0632477, 499.4380493, -205.3341675, 519.1251221, -717.1883545, 704.7721558
4: -568.3402710, 385.4368286, -590.9782715, 398.9001770, -967.2403564, 976.4151001

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8219832, upper bound: 808.4947826
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0029985, upper bound: 808.5811780
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -455.0569763, 380.6590271, -825.5484619, 827.3479004
1: -356.6080017, 360.9292603, -364.9006348, 369.0178833, -725.6258545, 725.8298950
2: -518.1353760, 393.8309326, -530.4962158, 402.6548462, -920.7902222, 924.3271484
3: -200.9437408, 507.3840332, -205.3341675, 519.1251221, -720.0687256, 712.7181396
4: -577.2812500, 390.2192383, -590.9782715, 398.9001770, -976.1813965, 981.1973877

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2538789, upper bound: 808.2641590
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5813265, upper bound: 808.5812322
time: 0.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.18 seconds
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -807.7105309, upper bound: 804.9129156
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -808.5187486, upper bound: 805.8381264
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2182480
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -808.5189495, upper bound: 806.2182480
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -808.8219832, upper bound: 808.4947826
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -809.0029985, upper bound: 808.5811780
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -808.2538789, upper bound: 808.2641590
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -808.5813265, upper bound: 808.5812322

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -451.4883728, 378.0201416, -243.2164001, 214.0600586, -665.5483398, 621.2365723
1: -362.0157776, 366.4407043, -193.1456299, 209.3964691, -571.4121704, 559.5862427
2: -526.2915039, 399.8591919, -278.1663513, 229.9753418, -756.2668457, 678.0255127
3: -203.8824615, 515.1088867, -117.0500183, 279.4069214, -483.2893677, 632.1589355
4: -586.3282471, 396.1748047, -311.8818665, 227.1875916, -813.5158691, 708.0566406

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3645107, upper bound: 804.9117821
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7074977, upper bound: 804.9122773
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -451.4883728, 378.0201416, -243.7677917, 213.4292908, -664.9176636, 621.7879028
1: -362.0157776, 366.4407043, -193.7469482, 208.7287445, -570.7445068, 560.1875610
2: -526.2915039, 399.8591919, -278.9052429, 229.1844482, -755.4759521, 678.7642212
3: -203.8824615, 515.1088867, -116.8389282, 279.3439636, -483.2264404, 631.9478149
4: -586.3282471, 396.1748047, -312.4699707, 226.3032074, -812.6314087, 708.6447754

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5182134, upper bound: 805.8380155
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2765259, upper bound: 805.5673919
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1759019, upper bound: 805.5677182
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -332.7971191, 281.7926941, -736.8496704, 713.4561157
1: -364.9006348, 369.0178833, -265.7110901, 274.8519287, -639.7525024, 634.7290039
2: -530.4962158, 402.6548462, -384.1240845, 300.9845886, -831.4808350, 786.7789307
3: -205.3341675, 519.1251221, -153.6954041, 379.7464600, -585.0805664, 672.8205566
4: -590.9782715, 398.9001770, -429.0441895, 297.0279236, -888.0062256, 827.9442139

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4183111, upper bound: 806.2179138
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2182480
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2182480
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -339.1815186, 285.2177429, -740.2747192, 719.8405762
1: -364.9006348, 369.0178833, -270.9281616, 278.1785583, -643.0792236, 639.9460449
2: -530.4962158, 402.6548462, -391.5458069, 304.4880066, -834.9841919, 794.2006836
3: -205.3341675, 519.1251221, -155.5240631, 386.5900574, -591.9240112, 674.6491089
4: -590.9782715, 398.9001770, -436.9592896, 300.3887939, -891.3670654, 835.8594360

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5184334, upper bound: 806.2179138
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5189495, upper bound: 806.2182480
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5189495, upper bound: 806.2182480
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -394.8950500, 334.1777039, -455.0569763, 380.6590271, -775.5540771, 789.2346802
1: -316.1249084, 323.8029480, -364.9006348, 369.0178833, -685.1428223, 688.7036133
2: -459.4748840, 353.5050354, -530.4962158, 402.6548462, -862.1297607, 884.0011597
3: -179.7761841, 450.4103699, -205.3341675, 519.1251221, -698.9011841, 655.7445068
4: -512.8085938, 350.7399292, -590.9782715, 398.9001770, -911.7086182, 941.7182007

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4793927, upper bound: 806.2223232
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8006439, upper bound: 808.3867761
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -673.0687256, 546.6938477, -448.7618103, 374.9576111, -1048.0260010, 991.7104492
1: -541.0684204, 531.0290527, -359.7861938, 363.5530396, -904.6214600, 887.2921143
2: -782.8427124, 578.2561035, -522.9698486, 396.9032898, -1179.7458496, 1097.5567627
3: -297.9616089, 757.8487549, -202.5752258, 511.6842651, -806.0300903, 960.4239502
4: -870.5672607, 571.6234131, -582.6387329, 393.1361694, -1263.7033691, 1150.8455811

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0021580, upper bound: 808.2540709
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0021580, upper bound: 808.5811780
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -400.8190918, 338.1311340, -455.0569763, 380.6590271, -781.4780884, 793.1881104
1: -321.0563354, 327.5602722, -364.9006348, 369.0178833, -690.0742188, 692.4609375
2: -466.4544678, 357.4834900, -530.4962158, 402.6548462, -869.1093140, 887.9797363
3: -182.1852417, 457.0790405, -205.3341675, 519.1251221, -701.3102417, 662.4130859
4: -520.2389526, 354.6555786, -590.9782715, 398.9001770, -919.1390991, 945.6337280

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2215378, upper bound: 806.2935063
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0856053, upper bound: 808.0986704
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -683.3941650, 554.1363525, -448.7618103, 374.9576111, -1058.3516846, 999.4845581
1: -549.4193726, 538.1006470, -359.7861938, 363.5530396, -912.9724121, 894.6148071
2: -794.7759399, 585.9496460, -522.9698486, 396.9032898, -1191.6790771, 1105.5305176
3: -302.0893250, 768.9462280, -202.5752258, 511.6842651, -810.3863525, 971.5214233
4: -883.6091919, 579.2677612, -582.6387329, 393.1361694, -1276.7451172, 1158.8063965

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5804861, upper bound: 808.2542117
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5804861, upper bound: 808.5812322
time: 1.25 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.51 seconds
IS_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -807.3645107, upper bound: 804.9117821
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -807.7074977, upper bound: 804.9122773
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -808.2765259, upper bound: 805.5673919
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -808.1759019, upper bound: 805.5677182
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2182480
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2182480
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -808.5189495, upper bound: 806.2182480
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -808.5189495, upper bound: 806.2182480
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -808.4793927, upper bound: 806.2223232
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -808.8006439, upper bound: 808.3867761
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -809.0021580, upper bound: 808.2540709
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -809.0021580, upper bound: 808.5811780
IS_A2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 3.51
Output dim: 4, lower bound: -806.2215378, upper bound: 806.2935063
IS_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -808.0856053, upper bound: 808.0986704
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -808.5804861, upper bound: 808.2542117
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.51
Output dim: 4, lower bound: -808.5804861, upper bound: 808.5812322

## BFS IS instance: IS_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -439.0681152, 367.6489563, -243.2164001, 214.0600586, -653.1279907, 610.8652344
1: -352.0338745, 356.5600586, -193.1456299, 209.3964691, -561.4301758, 549.7056885
2: -511.5484924, 389.2421265, -278.1663513, 229.9753418, -741.5237427, 667.4082031
3: -198.4695129, 501.1130371, -117.0500183, 279.4069214, -477.8764038, 618.1630249
4: -569.9522095, 385.4745483, -311.8818665, 227.1875916, -797.1397705, 697.3564453

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7074977, upper bound: 804.9122773
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7074977, upper bound: 804.9122773
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -321.8611450, 282.7077332, -243.4473877, 213.2043152, -535.0654297, 526.1551514
1: -257.4906921, 273.7072449, -193.4880829, 208.5100708, -466.0007324, 467.1953125
2: -375.1592712, 299.4261475, -278.5323792, 228.9493256, -604.1085815, 577.9584351
3: -151.2905121, 372.2079468, -116.7136536, 278.9945374, -430.2850342, 488.9216003
4: -418.4123840, 297.1170959, -312.0572815, 226.0706024, -644.4829102, 609.1743774

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2763910, upper bound: 805.5673195
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2355145, upper bound: 805.5670084
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0967135, upper bound: 805.5670369
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -439.0681152, 367.6489563, -243.7677917, 213.4292908, -652.4973145, 611.4166870
1: -352.0338745, 356.5600586, -193.7469482, 208.7287445, -560.7624512, 550.3070068
2: -511.5484924, 389.2421265, -278.9052429, 229.1844482, -740.7328491, 668.1470337
3: -198.4695129, 501.1130371, -116.8389282, 279.3439636, -477.8134155, 617.9519653
4: -569.9522095, 385.4745483, -312.4699707, 226.3032074, -796.2553711, 697.9445190

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1755727, upper bound: 805.5672492
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1754337, upper bound: 805.5670235
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1582334, upper bound: 805.5675451
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -332.7971191, 281.7926941, -719.5649414, 700.2384033
1: -350.6974182, 356.2439270, -265.7110901, 274.8519287, -625.5492554, 621.9550171
2: -509.7792053, 388.8671265, -384.1240845, 300.9845886, -810.7637329, 772.9910889
3: -198.0632477, 499.4380493, -153.6954041, 379.7464600, -577.8096924, 653.1334229
4: -568.3402710, 385.4368286, -429.0441895, 297.0279236, -865.3681641, 814.4809570

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4183111, upper bound: 806.2179096
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2179391
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2024902, upper bound: 806.2180104
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -332.7971191, 281.7926941, -726.6820068, 705.0880127
1: -356.6080017, 360.9292603, -265.7110901, 274.8519287, -631.4597778, 626.6403809
2: -518.1353760, 393.8309326, -384.1240845, 300.9845886, -819.1199951, 777.9550171
3: -200.9437408, 507.3840332, -153.6954041, 379.7464600, -580.6901855, 661.0794678
4: -577.2812500, 390.2192383, -429.0441895, 297.0279236, -874.3092041, 819.2633057

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4183111, upper bound: 806.2179138
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2180071
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2024902, upper bound: 806.2180104
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -437.7723999, 367.4412842, -339.1815186, 285.2177429, -722.9901123, 706.6228027
1: -350.6974182, 356.2439270, -270.9281616, 278.1785583, -628.8759766, 627.1720581
2: -509.7792053, 388.8671265, -391.5458069, 304.4880066, -814.2671509, 780.4129028
3: -198.0632477, 499.4380493, -155.5240631, 386.5900574, -584.6531372, 654.9620361
4: -568.3402710, 385.4368286, -436.9592896, 300.3887939, -868.7290649, 822.3961182

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4183111, upper bound: 806.2179096
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2179391
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2024074, upper bound: 806.2180104
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -444.8894653, 372.2908936, -339.1815186, 285.2177429, -730.1071777, 711.4724121
1: -356.6080017, 360.9292603, -270.9281616, 278.1785583, -634.7865601, 631.8574219
2: -518.1353760, 393.8309326, -391.5458069, 304.4880066, -822.6232910, 785.3767090
3: -200.9437408, 507.3840332, -155.5240631, 386.5900574, -587.5335693, 662.9080200
4: -577.2812500, 390.2192383, -436.9592896, 300.3887939, -877.6700439, 827.1785278

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4183111, upper bound: 806.2179138
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2180071
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2024074, upper bound: 806.2180104
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -391.8000488, 331.9727173, -361.1927795, 310.2428589, -702.0429077, 693.1655273
1: -313.6260376, 321.6509399, -288.8671265, 300.3380127, -613.9640503, 610.5178833
2: -455.8306580, 351.1705322, -419.7289124, 328.2561951, -784.0868530, 770.8993530
3: -178.5447388, 446.9456177, -166.5698853, 413.2271118, -591.7718506, 613.5155029
4: -508.7753601, 348.4544678, -468.4738770, 325.9736023, -834.7488403, 816.9282837

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4793927, upper bound: 806.2223232
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4793930, upper bound: 806.2223232
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -394.8950500, 334.1777039, -429.8060913, 359.7723999, -754.6674194, 763.9837646
1: -316.1249084, 323.8029480, -344.2817688, 349.0133667, -665.1383057, 668.0846558
2: -459.4748840, 353.5050354, -500.1039734, 381.3972168, -840.8720703, 853.6088867
3: -179.7761841, 450.4103699, -194.8598633, 489.6657715, -669.4417114, 645.2702637
4: -512.8085938, 350.7399292, -557.3035278, 377.4635315, -890.2720947, 908.0434570

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8004794, upper bound: 808.3867761
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8004794, upper bound: 808.3867718
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -672.0267334, 545.8126831, -410.5120544, 346.1290894, -1018.1558228, 952.6533203
1: -540.2261963, 530.1716919, -328.9624634, 335.2947083, -875.5208740, 855.6688232
2: -781.6167603, 577.3316650, -478.2442322, 365.9327087, -1147.5494385, 1051.9147949
3: -297.4959717, 756.6791992, -186.3758240, 468.3313599, -762.1388550, 943.0549927
4: -869.2042847, 570.7242432, -533.3006592, 362.9630737, -1232.1673584, 1100.6188965

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5912608, upper bound: 806.2231261
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9944219, upper bound: 808.0860963
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -673.0687256, 546.6938477, -709.0429688, 574.5297241, -1238.3945312, 1246.4310303
1: -541.0684204, 531.0290527, -570.2039185, 557.9644775, -1091.6247559, 1093.7711182
2: -782.8427124, 578.2561035, -825.2233887, 607.5036011, -1381.8475342, 1395.0394287
3: -297.9616089, 757.8487549, -313.2885437, 798.1068726, -1090.7478027, 1065.7365723
4: -870.5672607, 571.6234131, -917.2584229, 600.5452271, -1461.8409424, 1479.7684326

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5912611, upper bound: 806.2231261
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9944219, upper bound: 808.5197420
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -375.5244141, 316.8089294, -455.0569763, 380.6590271, -756.1834106, 771.8659058
1: -300.3884277, 307.2716675, -364.9006348, 369.0178833, -669.4063110, 672.1723022
2: -436.0256958, 335.8262939, -530.4962158, 402.6548462, -838.6805420, 866.3225098
3: -171.7265167, 427.6165771, -205.3341675, 519.1251221, -690.8516235, 632.9505615
4: -486.6319275, 332.6908569, -590.9782715, 398.9001770, -885.5319824, 923.6689453

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0852689, upper bound: 808.0852631
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0852689, upper bound: 808.0986656
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -682.4240723, 553.3163452, -410.5120544, 346.1290894, -1028.5532227, 960.4850464
1: -548.6352539, 537.3026123, -328.9624634, 335.2947083, -883.9299316, 863.0480347
2: -793.6350708, 585.0892944, -478.2442322, 365.9327087, -1159.5676270, 1059.9486084
3: -301.6557922, 767.8592529, -186.3758240, 468.3313599, -766.5247192, 954.2351074
4: -882.3404541, 578.4313965, -533.3006592, 362.9630737, -1245.3033447, 1108.6387939

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2215378, upper bound: 807.3088474
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5189385, upper bound: 808.0861699
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -683.3941650, 554.1363525, -709.0429688, 574.5297241, -1248.1868896, 1254.2048340
1: -549.4193726, 538.1006470, -570.2039185, 557.9644775, -1099.5979004, 1101.0936279
2: -794.7759399, 585.9496460, -825.2233887, 607.5036011, -1393.2791748, 1403.0131836
3: -302.0893250, 768.9462280, -313.2885437, 798.1068726, -1095.1038818, 1076.6745605
4: -883.6091919, 579.2677612, -917.2584229, 600.5452271, -1474.3646240, 1487.7293701

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2215378, upper bound: 807.3088474
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5189385, upper bound: 808.5197818
time: 0.68 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.19 seconds
IS_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -807.7074977, upper bound: 804.9122773
IS_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -807.7074977, upper bound: 804.9122773
IS_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.2355145, upper bound: 805.5670084
IS_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.0967135, upper bound: 805.5670369
IS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.1754337, upper bound: 805.5670235
IS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.1582334, upper bound: 805.5675451
IS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2179391
IS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.2024902, upper bound: 806.2180104
IS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2180071
IS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.2024902, upper bound: 806.2180104
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2179391
IS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.2024074, upper bound: 806.2180104
IS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.4191623, upper bound: 806.2180071
IS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.2024074, upper bound: 806.2180104
IS_A2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.4793927, upper bound: 806.2223232
IS_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.4793930, upper bound: 806.2223232
IS_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.8004794, upper bound: 808.3867761
IS_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.8004794, upper bound: 808.3867718
IS_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.5912608, upper bound: 806.2231261
IS_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9944219, upper bound: 808.0860963
IS_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.5912611, upper bound: 806.2231261
IS_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9944219, upper bound: 808.5197420
IS_A2_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.0852689, upper bound: 808.0852631
IS_A2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.0852689, upper bound: 808.0986656
IS_A2_B2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 4, lower bound: -806.2215378, upper bound: 807.3088474
IS_A2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.5189385, upper bound: 808.0861699
IS_A2_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 4, lower bound: -806.2215378, upper bound: 807.3088474
IS_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.5189385, upper bound: 808.5197818

## BFS IS instance: IS_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -420.7584229, 353.7670898, -243.2164001, 214.0600586, -634.8183594, 596.9835205
1: -337.0100708, 343.1326599, -193.1456299, 209.3964691, -546.4064331, 536.2781982
2: -489.6396484, 374.7192688, -278.1663513, 229.9753418, -719.6148682, 652.8856201
3: -190.8235931, 480.2705383, -117.0500183, 279.4069214, -470.2304993, 597.3205566
4: -546.0025024, 371.3430481, -311.8818665, 227.1875916, -773.1900635, 683.2249146

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6628989, upper bound: 804.6419642
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7018100, upper bound: 804.6544399
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 20

Time for candidate selection: 8.44 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5558757, upper bound: 804.5749082
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6957685, upper bound: 804.9122773
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7063242, upper bound: 804.8266699
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5087058, upper bound: 804.1237117
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6072178, upper bound: 802.2131202
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7074977, upper bound: 804.9122773
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -428.4158630, 358.9230347, -243.2164001, 214.0600586, -642.4758911, 602.1393433
1: -343.3513794, 348.1182861, -193.1456299, 209.3964691, -552.7478027, 541.2638550
2: -498.6167603, 380.1333618, -278.1663513, 229.9753418, -728.5921021, 658.2996216
3: -193.8960724, 488.8001709, -117.0500183, 279.4069214, -473.3029785, 605.8502197
4: -555.6296387, 376.4710083, -311.8818665, 227.1875916, -782.8171997, 688.3529053

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6628989, upper bound: 804.6419642
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7018100, upper bound: 804.6544399
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 20

Time for candidate selection: 8.25 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5558757, upper bound: 804.5749082
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6957685, upper bound: 804.9122773
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7063242, upper bound: 804.8266699
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5087058, upper bound: 804.1237117
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6072178, upper bound: 802.2131202
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7074977, upper bound: 804.9122773
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -268.3532104, 237.5550232, -238.3162079, 209.2145844, -477.5678101, 475.8711853
1: -214.4407349, 230.2907562, -189.3424225, 204.6557007, -419.0964355, 419.6331787
2: -312.2075500, 251.9739685, -272.5277405, 224.7811584, -536.9887085, 524.5017090
3: -126.2927704, 311.3432922, -114.4921188, 273.2216492, -399.5144043, 425.8354187
4: -347.8852844, 249.9773102, -305.3797607, 221.9424591, -569.8276978, 555.3569946

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.9215585, upper bound: 805.5662128
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.9215585, upper bound: 805.5662128
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -315.2450867, 277.0806274, -243.4473877, 213.2043152, -528.4493408, 520.5280151
1: -252.1258392, 268.2949524, -193.4880829, 208.5100708, -460.6358643, 461.7830200
2: -367.3490906, 293.6072388, -278.5323792, 228.9493256, -596.2984009, 572.1395874
3: -148.3249817, 364.5948792, -116.7136536, 278.9945374, -427.3195190, 481.3085022
4: -409.7304382, 291.3341980, -312.0572815, 226.0706024, -635.8009644, 603.3914185

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0964348, upper bound: 805.5668326
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8375444, upper bound: 805.5662993
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8375444, upper bound: 805.5670369
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -372.9372864, 313.0272217, -238.6308594, 209.4359131, -582.3731689, 551.6580811
1: -298.8602295, 303.8649597, -189.5968323, 204.8708801, -503.7311096, 493.4617920
2: -433.7326355, 332.0969849, -272.8941040, 225.0126953, -658.7453613, 604.9910889
3: -168.7735291, 425.8609924, -114.6157455, 273.5653076, -442.3388062, 540.4767456
4: -482.6667175, 328.5703125, -305.7852173, 222.1712952, -704.8379517, 634.3555298

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1751146, upper bound: 805.5668632
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1749112, upper bound: 805.2828694
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1750557, upper bound: 805.5670235
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1750557, upper bound: 805.5670235
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -432.5902100, 361.9089661, -243.7677917, 213.4292908, -646.0195312, 605.6767578
1: -346.7731934, 351.0227356, -193.7469482, 208.7287445, -555.5017090, 544.7695923
2: -503.8969421, 383.3670349, -278.9052429, 229.1844482, -733.0814209, 662.2720947
3: -195.4917145, 493.7084351, -116.8389282, 279.3439636, -474.8356323, 610.5473633
4: -561.5106201, 379.5962524, -312.4699707, 226.3032074, -787.8138428, 692.0661011

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1579437, upper bound: 805.5668605
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7074977, upper bound: 805.5675451
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7074977, upper bound: 805.5563515
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -362.6666565, 305.9226074, -327.1813660, 277.3379822, -640.0045166, 633.1038818
1: -290.2025757, 296.8375549, -261.1712952, 270.5529480, -560.7554932, 558.0088501
2: -421.4447021, 324.4236450, -377.5047913, 296.3498230, -717.7944946, 701.9284058
3: -164.7423096, 413.9035950, -151.1579590, 373.3670654, -538.1093140, 565.0615234
4: -469.5854187, 321.2560120, -421.6633911, 292.4135742, -761.9990234, 742.9193726

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9284332, upper bound: 805.9774825
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.6743763, upper bound: 806.1965887
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9284490, upper bound: 806.2176570
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3765261, upper bound: 806.2167334
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8570264, upper bound: 806.2169812
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -428.7608032, 359.9247742, -332.7971191, 281.7926941, -710.5533447, 692.7219238
1: -343.3657837, 348.9992065, -265.7110901, 274.8519287, -618.2175903, 614.7103271
2: -499.1161499, 381.0668945, -384.1240845, 300.9845886, -800.1005859, 765.1909180
3: -194.1269379, 489.1353455, -153.6954041, 379.7464600, -573.8734131, 642.8307495
4: -556.5690918, 377.6802368, -429.0441895, 297.0279236, -853.5970459, 806.7243042

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9744703, upper bound: 806.2175071
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5111451, upper bound: 805.0740386
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9409751, upper bound: 806.2182484
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -380.1496887, 318.2424622, -327.1813660, 277.3379822, -657.4875488, 645.4237061
1: -304.5241394, 308.7597961, -261.1712952, 270.5529480, -575.0770874, 569.9310913
2: -441.9195862, 337.3076477, -377.5047913, 296.3498230, -738.2694092, 714.8123169
3: -171.6442566, 433.2800598, -151.1579590, 373.3670654, -545.0111084, 584.4379883
4: -491.8168945, 333.9521179, -421.6633911, 292.4135742, -784.2304688, 755.6154785

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0615881, upper bound: 806.2168984
time: 1.04 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0615930, upper bound: 806.2180071
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -436.6898499, 365.1800842, -332.7971191, 281.7926941, -718.4825439, 697.9771729
1: -349.9454956, 354.0716248, -265.7110901, 274.8519287, -624.7973022, 619.7827148
2: -508.4296265, 386.5229187, -384.1240845, 300.9845886, -809.4141846, 770.6468506
3: -197.2324371, 497.9155884, -153.6954041, 379.7464600, -576.9788208, 651.6109619
4: -566.5614014, 382.8739319, -429.0441895, 297.0279236, -863.5893555, 811.9179688

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019481, upper bound: 806.2172974
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0267091, upper bound: 806.2170470
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0267091, upper bound: 806.2180104
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -362.6666565, 305.9226074, -334.0037537, 281.0453186, -643.7119141, 639.9263916
1: -290.2025757, 296.8375549, -266.7456055, 274.1578979, -564.3604736, 563.5831299
2: -421.4447021, 324.4236450, -385.4673462, 300.1467285, -721.5914307, 709.8909302
3: -164.7423096, 413.9035950, -153.1459808, 380.6857300, -545.4280396, 567.0495605
4: -469.5854187, 321.2560120, -430.1876221, 296.0820312, -765.6674805, 751.4436035

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9284290, upper bound: 805.9774825
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9281914, upper bound: 805.7845162
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -428.7608032, 359.9247742, -339.1815186, 285.2177429, -713.9785156, 699.1063232
1: -343.3657837, 348.9992065, -270.9281616, 278.1785583, -621.5443115, 619.9273682
2: -499.1161499, 381.0668945, -391.5458069, 304.4880066, -803.6039429, 772.6126709
3: -194.1269379, 489.1353455, -155.5240631, 386.5900574, -580.7168579, 644.6593018
4: -556.5690918, 377.6802368, -436.9592896, 300.3887939, -856.9578857, 814.6395264

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9744600, upper bound: 806.2175071
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.4654196, upper bound: 806.2176848
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9744530, upper bound: 806.2179755
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -380.1496887, 318.2424622, -334.0037537, 281.0453186, -661.1948853, 652.2462158
1: -304.5241394, 308.7597961, -266.7456055, 274.1578979, -578.6820068, 575.5053711
2: -441.9195862, 337.3076477, -385.4673462, 300.1467285, -742.0662842, 722.7748413
3: -171.6442566, 433.2800598, -153.1459808, 380.6857300, -552.3299561, 586.4260254
4: -491.8168945, 333.9521179, -430.1876221, 296.0820312, -787.8989258, 764.1397705

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2221001, upper bound: 806.2168984
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2221001, upper bound: 806.2180071
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -436.6898499, 365.1800842, -339.1815186, 285.2177429, -721.9075928, 704.3615723
1: -349.9454956, 354.0716248, -270.9281616, 278.1785583, -628.1240234, 624.9997559
2: -508.4296265, 386.5229187, -391.5458069, 304.4880066, -812.9176025, 778.0686646
3: -197.2324371, 497.9155884, -155.5240631, 386.5900574, -583.8222046, 653.4395752
4: -566.5614014, 382.8739319, -436.9592896, 300.3887939, -866.9501953, 819.8331909

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019378, upper bound: 806.2172974
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0272296, upper bound: 806.2170470
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0272296, upper bound: 806.2180104
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -391.8000488, 331.9727173, -322.1797180, 280.6486816, -672.4487305, 654.1524658
1: -313.6260376, 321.6509399, -257.3641052, 271.4838867, -585.1099243, 579.0148315
2: -455.8306580, 351.1705322, -373.8704529, 296.7969971, -752.6276855, 725.0408936
3: -178.5447388, 446.9456177, -150.5571747, 368.8250732, -547.3697510, 597.5028076
4: -508.7753601, 348.4544678, -417.9834900, 295.1875305, -803.9628296, 766.4379883

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4793326, upper bound: 806.2223232
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7237672, upper bound: 806.2165058
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4384909, upper bound: 806.2168268
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -391.8000488, 331.9727173, -599.7645874, 485.5643005, -873.4865723, 931.7373047
1: -313.6260376, 321.6509399, -482.1784668, 471.1179199, -781.2471924, 803.8294067
2: -455.8306580, 351.1705322, -697.2562866, 513.9586182, -965.9360962, 1048.4267578
3: -178.5447388, 446.9456177, -266.1380615, 673.0794067, -851.6241455, 709.2142334
4: -508.7753601, 348.4544678, -775.3367310, 508.6732483, -1013.8529053, 1123.7912598

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_A1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7237672, upper bound: 806.2170154
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_A1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4384912, upper bound: 806.2173365
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -394.8950500, 334.1777039, -411.3385620, 345.7905273, -740.6855469, 745.5162354
1: -316.1249084, 323.8029480, -329.1507263, 335.5002441, -651.6251221, 652.9535522
2: -459.4748840, 353.5050354, -478.0440063, 366.8049011, -826.2797852, 831.5488892
3: -179.7761841, 450.4103699, -187.2203827, 468.7021179, -648.4782104, 637.6307373
4: -512.8085938, 350.7399292, -533.1908569, 363.2216187, -876.0301514, 883.9307861

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7998731, upper bound: 808.0852824
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7998731, upper bound: 808.3867761
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -394.8950500, 334.1777039, -419.5442810, 351.3311157, -746.2261963, 753.7219849
1: -316.1249084, 323.8029480, -335.8997192, 340.8306885, -656.9555664, 659.7026367
2: -459.4748840, 353.5050354, -487.5963440, 372.4834900, -831.9583130, 841.1012573
3: -179.7761841, 450.4103699, -190.3561554, 477.7799683, -657.5561523, 640.7664795
4: -512.8085938, 350.7399292, -543.4680786, 368.7157288, -881.5242310, 894.2080078

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8002456, upper bound: 808.3867718
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7999405, upper bound: 808.2022609
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -668.9580688, 543.4990845, -322.1797180, 280.6486816, -949.6067505, 861.1368408
1: -537.7564087, 527.9258423, -257.3641052, 271.4838867, -809.2402954, 781.1916504
2: -778.0234985, 574.9021606, -373.8704529, 296.7969971, -1074.8204346, 944.1097412
3: -296.2623596, 753.2307739, -150.5571747, 368.8250732, -660.7708740, 903.7878418
4: -865.2167358, 568.3463745, -417.9834900, 295.1875305, -1160.4041748, 981.7672119

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7683822, upper bound: 806.0278630
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7683825, upper bound: 806.0276854
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -671.5892944, 545.4428711, -384.8684692, 324.4189758, -996.0082397, 926.9279175
1: -539.8725586, 529.8118896, -308.0199585, 314.6398315, -854.5123901, 834.5657959
2: -781.1023560, 576.9437866, -447.4355774, 343.8702698, -1124.9726562, 1021.1289673
3: -297.3005371, 756.1882324, -175.7989807, 438.4773254, -732.2606201, 931.9871826
4: -868.6321411, 570.3469238, -499.2395325, 340.5835876, -1209.2154541, 1066.6640625

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9287562, upper bound: 808.0857035
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9747715, upper bound: 808.0855377
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -670.1503906, 544.5066528, -617.5979614, 504.8901062, -1165.3065186, 1152.2241211
1: -538.7200928, 528.9063721, -496.5155334, 490.3788452, -1021.2247314, 1017.4718018
2: -779.4259033, 575.9591675, -718.1162720, 534.5359497, -1304.8962402, 1284.7158203
3: -296.7947998, 754.5685425, -276.2371521, 694.8546143, -985.9476318, 1024.8012695
4: -866.7762451, 569.3744507, -798.5817871, 528.8070679, -1385.7259521, 1357.9113770

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5753250, upper bound: 806.2225220
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5661660, upper bound: 805.9367839
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8022853, upper bound: 806.1882555
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8036233, upper bound: 804.4869140
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -673.0687256, 546.6938477, -684.3961182, 554.4020386, -1218.0407715, 1221.8725586
1: -541.0684204, 531.0290527, -550.2756348, 538.5758667, -1071.9620361, 1073.8927002
2: -782.8427124, 578.2561035, -796.0435181, 586.5910645, -1360.6922607, 1366.0241699
3: -297.9616089, 757.8487549, -302.5914307, 770.0386963, -1062.7469482, 1055.1851807
4: -870.5672607, 571.6234131, -884.9140625, 579.6287842, -1440.8015137, 1447.6231689

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9288369, upper bound: 808.2027035
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9748523, upper bound: 808.2025377
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -375.5244141, 316.8089294, -410.5120544, 346.1290894, -721.6535034, 727.3208008
1: -300.3884277, 307.2716675, -328.9624634, 335.2947083, -635.6831055, 636.2340698
2: -436.0256958, 335.8262939, -478.2442322, 365.9327087, -801.9583740, 814.0704956
3: -171.7265167, 427.6165771, -186.3758240, 468.3313599, -640.0577393, 613.9923096
4: -486.6319275, 332.6908569, -533.3006592, 362.9630737, -849.5949707, 865.9915161

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0852046, upper bound: 808.0852631
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0852051, upper bound: 808.0852631
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -375.5244141, 316.8089294, -707.5430908, 573.2620239, -945.0330811, 1024.3520508
1: -300.3884277, 307.2716675, -568.9910889, 556.7310181, -853.5438843, 876.2626343
2: -436.0256958, 335.8262939, -823.4586182, 606.1734619, -1038.4305420, 1159.2849121
3: -171.7265167, 427.6165771, -312.6175842, 796.4243774, -968.1508789, 736.5638428
4: -486.6319275, 332.6908569, -915.2965698, 599.2559204, -1082.4161377, 1247.9874268

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0851100, upper bound: 808.0212536
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0851419, upper bound: 808.0985339
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -658.2756348, 533.6447144, -410.5120544, 346.1290894, -1004.4047241, 940.6156006
1: -529.1301880, 518.3414307, -328.9624634, 335.2947083, -864.4249268, 843.8519287
2: -765.0959473, 564.6002808, -478.2442322, 365.9327087, -1131.0286865, 1039.2618408
3: -291.1478882, 740.4421387, -186.3758240, 468.3313599, -756.1654663, 926.8179321
4: -850.6744995, 557.9743042, -533.3006592, 362.9630737, -1213.6375732, 1088.0334473

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4763058, upper bound: 808.0859158
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022472, upper bound: 808.0853295
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -659.6030273, 534.7671509, -709.0429688, 574.5297241, -1224.4503174, 1234.6025391
1: -530.2028198, 519.4345093, -570.2039185, 557.9644775, -1080.4262695, 1082.1643066
2: -766.6569824, 565.7820435, -825.2233887, 607.5036011, -1365.3070068, 1382.6118164
3: -291.7438049, 741.9286499, -313.2885437, 798.1068726, -1084.8847656, 1049.7211914
4: -852.4110718, 559.1191406, -917.2584229, 600.5452271, -1443.3438721, 1467.3944092

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023029, upper bound: 808.2162314
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023048, upper bound: 808.2023301
time: 0.76 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.38 seconds
IS_A2_B1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -807.6072178, upper bound: 802.2131202
IS_A2_B1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -807.7074977, upper bound: 804.9122773
IS_A2_B1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -807.6072178, upper bound: 802.2131202
IS_A2_B1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -807.7074977, upper bound: 804.9122773
IS_A2_B1_B1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -805.9215585, upper bound: 805.5662128
IS_A2_B1_B1_B2_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -805.9215585, upper bound: 805.5662128
IS_A2_B1_B1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -805.8375444, upper bound: 805.5662993
IS_A2_B1_B1_B2_A1_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -805.8375444, upper bound: 805.5670369
IS_A2_B1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.1750557, upper bound: 805.5670235
IS_A2_B1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.1750557, upper bound: 805.5670235
IS_A2_B1_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -807.7074977, upper bound: 805.5675451
IS_A2_B1_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -807.7074977, upper bound: 805.5563515
IS_A2_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.3765261, upper bound: 806.2167334
IS_A2_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.8570264, upper bound: 806.2169812
IS_A2_B1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.5111451, upper bound: 805.0740386
IS_A2_B1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.9409751, upper bound: 806.2182484
IS_A2_B1_B2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -806.0615881, upper bound: 806.2168984
IS_A2_B1_B2_B1_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -806.0615930, upper bound: 806.2180071
IS_A2_B1_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -806.0267091, upper bound: 806.2170470
IS_A2_B1_B2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -806.0267091, upper bound: 806.2180104
IS_A2_B1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.9284290, upper bound: 805.9774825
IS_A2_B1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.9281914, upper bound: 805.7845162
IS_A2_B1_B2_B2_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -805.4654196, upper bound: 806.2176848
IS_A2_B1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.9744530, upper bound: 806.2179755
IS_A2_B1_B2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -806.2221001, upper bound: 806.2168984
IS_A2_B1_B2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -806.2221001, upper bound: 806.2180071
IS_A2_B1_B2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -806.0272296, upper bound: 806.2170470
IS_A2_B1_B2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -806.0272296, upper bound: 806.2180104
IS_A2_B2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -807.7237672, upper bound: 806.2165058
IS_A2_B2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.4384909, upper bound: 806.2168268
IS_A2_B2_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -807.7237672, upper bound: 806.2170154
IS_A2_B2_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.4384912, upper bound: 806.2173365
IS_A2_B2_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.7998731, upper bound: 808.0852824
IS_A2_B2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.7998731, upper bound: 808.3867761
IS_A2_B2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.8002456, upper bound: 808.3867718
IS_A2_B2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.7999405, upper bound: 808.2022609
IS_A2_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.7683822, upper bound: 806.0278630
IS_A2_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.7683825, upper bound: 806.0276854
IS_A2_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.9287562, upper bound: 808.0857035
IS_A2_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.9747715, upper bound: 808.0855377
IS_A2_B2_A1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -806.8022853, upper bound: 806.1882555
IS_A2_B2_A1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 4, lower bound: -806.8036233, upper bound: 804.4869140
IS_A2_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.9288369, upper bound: 808.2027035
IS_A2_B2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.9748523, upper bound: 808.2025377
IS_A2_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.0852046, upper bound: 808.0852631
IS_A2_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.0852051, upper bound: 808.0852631
IS_A2_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.0851100, upper bound: 808.0212536
IS_A2_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.0851419, upper bound: 808.0985339
IS_A2_B2_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.4763058, upper bound: 808.0859158
IS_A2_B2_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.2022472, upper bound: 808.0853295
IS_A2_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.2023029, upper bound: 808.2162314
IS_A2_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 4, lower bound: -808.2023048, upper bound: 808.2023301

## BFS IS instance: IS_A2_B1_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -420.7584229, 353.7670898, -238.2354889, 209.8302307, -630.5886230, 592.0025024
1: -337.0100708, 343.1326599, -189.1135406, 205.1921082, -542.2021484, 532.2461548
2: -489.6396484, 374.7192688, -272.6571655, 225.3511810, -714.9908447, 647.3764648
3: -190.8235931, 480.2705383, -114.4669647, 273.1453857, -463.9689941, 594.7374878
4: -546.0025024, 371.3430481, -305.8312378, 222.5730591, -768.5755005, 677.1742554

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3101254, upper bound: 802.0991867
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9154994, upper bound: 802.2131508
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4928063, upper bound: 801.5274835
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5517868, upper bound: 802.1762069
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4518495, upper bound: 802.1645957
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5393720, upper bound: 802.1645663
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -420.7584229, 353.7670898, -239.5700531, 211.6490021, -632.4074097, 593.3370972
1: -337.0100708, 343.1326599, -190.1708221, 207.0041809, -544.0142822, 533.3034668
2: -489.6396484, 374.7192688, -273.9182434, 227.3784790, -717.0180054, 648.6374512
3: -190.8235931, 480.2705383, -115.6930695, 275.5813293, -466.4049072, 595.9636230
4: -546.0025024, 371.3430481, -307.2666016, 224.6373596, -770.6398926, 678.6096191

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.4083494, upper bound: 804.5756713
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0574757, upper bound: 804.9127805
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6219244, upper bound: 804.7336665
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5631595, upper bound: 804.0962507
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5071420, upper bound: 804.8996177
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5074500, upper bound: 804.8603094
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -428.4158630, 358.9230347, -238.2354889, 209.8302307, -638.2460938, 597.1583252
1: -343.3513794, 348.1182861, -189.1135406, 205.1921082, -548.5434570, 537.2317505
2: -498.6167603, 380.1333618, -272.6571655, 225.3511810, -723.9679565, 652.7905273
3: -193.8960724, 488.8001709, -114.4669647, 273.1453857, -467.0414429, 603.2671509
4: -555.6296387, 376.4710083, -305.8312378, 222.5730591, -778.2025146, 682.3022461

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.4471505, upper bound: 802.0988712
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5855497, upper bound: 802.2131202
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3810083, upper bound: 801.5267680
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6061145, upper bound: 802.1756254
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.1527134, upper bound: 802.1738252
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -428.4158630, 358.9230347, -239.5700531, 211.6490021, -640.0648804, 598.4929810
1: -343.3513794, 348.1182861, -190.1708221, 207.0041809, -550.3555908, 538.2891235
2: -498.6167603, 380.1333618, -273.9182434, 227.3784790, -725.9952393, 654.0515137
3: -193.8960724, 488.8001709, -115.6930695, 275.5813293, -469.4773560, 604.4932251
4: -555.6296387, 376.4710083, -307.2666016, 224.6373596, -780.2669678, 683.7376099

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5558757, upper bound: 804.5749082
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6957685, upper bound: 804.9122773
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7063242, upper bound: 804.7327043
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.1527233, upper bound: 804.7305592
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -372.9372864, 313.0272217, -125.4913940, 129.1822052, -502.1194153, 438.5186157
1: -298.8602295, 303.8649597, -98.8955765, 126.6194000, -425.4796143, 402.7604980
2: -433.7326355, 332.0969849, -142.2126617, 140.6814270, -574.4140625, 474.3096008
3: -168.7735291, 425.8609924, -70.7910309, 151.2403717, -320.0138855, 496.6520081
4: -482.6667175, 328.5703125, -160.3479462, 138.8716278, -621.5382690, 488.9181824

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1749962, upper bound: 805.5668632
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1747916, upper bound: 805.2828694
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1707310, upper bound: 805.1662763
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 20

Time for candidate selection: 9.64 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0574034, upper bound: 805.3086412
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1735816, upper bound: 805.2771134
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1747755, upper bound: 805.4686242
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1750557, upper bound: 805.5670235
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -372.9372864, 313.0272217, -228.8543701, 202.7087097, -575.6459351, 541.8815918
1: -298.8602295, 303.8649597, -181.7698517, 198.5247040, -497.3849487, 485.6347961
2: -433.7326355, 332.0969849, -261.3042908, 218.1866913, -651.9193115, 593.4011841
3: -168.7735291, 425.8609924, -110.9452515, 263.3286438, -432.1021423, 536.8062134
4: -482.6667175, 328.5703125, -292.9634705, 215.3675385, -698.0342407, 621.5337524

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1749962, upper bound: 805.5668632
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1747916, upper bound: 805.2828694
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1707310, upper bound: 805.1719839
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 20

Time for candidate selection: 9.28 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0574034, upper bound: 805.3086412
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1735816, upper bound: 805.2771133
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1245080, upper bound: 804.9104819
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1747755, upper bound: 805.4966888
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1750557, upper bound: 805.5670235
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -413.5912476, 347.7433472, -243.7677917, 213.4292908, -627.0205078, 591.5111084
1: -331.1743469, 337.3310547, -193.7469482, 208.7287445, -539.9030762, 531.0780029
2: -481.1799622, 368.5282288, -278.9052429, 229.1844482, -710.3643799, 647.4334106
3: -187.6909027, 472.1610107, -116.8389282, 279.3439636, -467.0347595, 588.9998169
4: -536.6521606, 365.1354980, -312.4699707, 226.3032074, -762.9553833, 677.6054688

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6628989, upper bound: 805.2832090
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7018075, upper bound: 805.1735886
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 43

Time for candidate selection: 8.14 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5558757, upper bound: 805.3090548
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7063242, upper bound: 805.2775346
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6957652, upper bound: 805.5573244
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5087058, upper bound: 804.9108937
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.1700945, upper bound: 805.5673668
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7074977, upper bound: 805.5675451
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -421.8020935, 353.0725708, -243.7677917, 213.4292908, -635.2313843, 596.8401489
1: -337.9725342, 342.4955139, -193.7469482, 208.7287445, -546.7012329, 536.2423706
2: -490.7892761, 374.1408997, -278.9052429, 229.1844482, -719.9737549, 653.0460205
3: -190.8523712, 481.2301636, -116.8389282, 279.3439636, -470.1963501, 598.0690918
4: -546.9939575, 370.5241394, -312.4699707, 226.3032074, -773.2971191, 682.9939575

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6628989, upper bound: 805.2222133
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7018075, upper bound: 805.0809649
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 43

Time for candidate selection: 8.68 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5558757, upper bound: 805.2933161
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6957652, upper bound: 805.5563455
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7063242, upper bound: 805.2330909
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5087058, upper bound: 804.6285144
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6602544, upper bound: 805.0977326
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6607812, upper bound: 805.5560615
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -238.8508759, 215.9740753, -326.8363037, 277.0902710, -515.9411621, 542.8103638
1: -190.3831024, 209.4111938, -260.8926392, 270.3126831, -460.6957397, 470.3038330
2: -277.0705872, 229.3539886, -377.0998230, 296.0925293, -573.1630859, 606.4537354
3: -114.6507111, 278.3254089, -151.0160370, 372.9862366, -487.6369019, 429.3414307
4: -309.5355530, 227.6612244, -421.2131653, 292.1578064, -601.6933594, 648.8743896

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3757389, upper bound: 805.9763155
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.1513023, upper bound: 806.1953685
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3756837, upper bound: 806.2166777
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 35

Time for candidate selection: 9.07 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3650921, upper bound: 804.7727098
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3757135, upper bound: 805.6219304
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3765261, upper bound: 806.2167334
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3738756, upper bound: 805.8587062
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3765261, upper bound: 806.2167334
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -350.1861877, 296.3976135, -327.1813660, 277.3379822, -627.5241699, 623.5787354
1: -280.1870117, 287.8131714, -261.1712952, 270.5529480, -550.7399902, 548.9844360
2: -406.6942444, 314.7042542, -377.5047913, 296.3498230, -703.0440674, 692.2090454
3: -159.5774536, 400.3084106, -151.1579590, 373.3670654, -532.9444580, 551.4663696
4: -453.2059937, 311.4501648, -421.6633911, 292.4135742, -745.6195068, 733.1132812

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8565864, upper bound: 805.9765633
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.4761813, upper bound: 806.1956177
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8565950, upper bound: 806.2169812
time: 1.02 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 43

Time for candidate selection: 9.46 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8546056, upper bound: 804.9012531
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8513258, upper bound: 806.2169812
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8563446, upper bound: 805.6221537
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=1011.34521484375
rel_dist={4: [-809.0067385931752, 809.0067385931752]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

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
- Time for IS candidates: 1.43 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.43
Output dim: 4, lower bound: -806.4689778, upper bound: 808.2251898
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.43
Output dim: 4, lower bound: -809.0037620, upper bound: 809.0037626

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -355.7917175, 298.0696411, -448.3735046, 376.2344360, -732.0261230, 746.4429932
1: -284.4114380, 290.6557922, -359.5960388, 364.8071289, -649.2185669, 650.2517700
2: -411.3229675, 317.9904480, -522.5211792, 398.0231934, -809.3461304, 840.5115967
3: -162.9068451, 405.6134033, -203.0482941, 512.1327515, -675.0396118, 608.6616821
4: -458.9186401, 313.6753845, -582.0831299, 394.3340454, -853.2526855, 895.7585449

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.3711909, upper bound: 808.0391165
time: 0.68 seconds

## Relational analysis of IS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8380044, upper bound: 807.7354474
time: 0.71 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2182824, upper bound: 808.1344875
time: 0.67 seconds

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
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2251898, upper bound: 806.4689778
time: 1.02 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2251898, upper bound: 806.4689778
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.55 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 4, lower bound: -805.8380044, upper bound: 807.7354474
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 4, lower bound: -806.2182824, upper bound: 808.1344875
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 4, lower bound: -808.2251898, upper bound: 806.4689778
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 4, lower bound: -808.2251898, upper bound: 806.4689778

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -251.5948334, 219.6874542, -421.1297913, 356.0421753, -607.6370239, 640.8172607
1: -200.1122742, 214.7951965, -337.5599365, 345.0800781, -545.1923218, 552.3550415
2: -288.3356018, 235.8116150, -490.3752136, 376.6383667, -664.9738770, 726.1868286
3: -120.3001709, 288.3111572, -191.9281006, 481.3605042, -601.6606445, 480.2392578
4: -322.8778687, 232.7944031, -546.5302124, 373.4487610, -696.3266602, 779.3245850

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8372757, upper bound: 807.7353988
time: 0.92 seconds

## Relational analysis of IS_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8369742, upper bound: 806.2033746
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8369742, upper bound: 807.7354474
time: 0.66 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -348.0386353, 292.4441223, -446.0192871, 374.1922302, -722.2308350, 738.4633789
1: -278.1151123, 285.1520081, -357.6835632, 362.8654785, -640.9805908, 642.8353882
2: -402.1842957, 312.0704346, -519.7103271, 396.0528564, -798.2370605, 831.7805176
3: -159.7416534, 396.7953796, -202.0726776, 509.3912659, -669.1328735, 598.8680420
4: -448.7807617, 307.8369446, -578.9666748, 392.2512817, -841.0320435, 886.8035889

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9386386, upper bound: 807.8293937
time: 0.69 seconds

## Relational analysis of IS_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2179647, upper bound: 807.6027596
time: 0.77 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2179647, upper bound: 807.9628608
time: 0.62 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -355.7917175, 298.0696411, -753.1265869, 736.4506836
1: -364.9006348, 369.0178833, -284.4114380, 290.6557922, -655.5563965, 653.4293213
2: -530.4962158, 402.6548462, -411.3229675, 317.9904480, -848.4865723, 813.9777832
3: -205.3341675, 519.1251221, -162.9068451, 405.6134033, -610.9473877, 682.0319824
4: -590.9782715, 398.9001770, -458.9186401, 313.6753845, -904.6536865, 857.8186646

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0391151, upper bound: 806.3711909
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7354458, upper bound: 805.8380044
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1344830, upper bound: 806.2182824
time: 0.66 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -455.0569763, 380.6590271, -455.0569763, 380.6590271, -835.7160034, 835.7160034
1: -364.9006348, 369.0178833, -364.9006348, 369.0178833, -733.9185181, 733.9185181
2: -530.4962158, 402.6548462, -530.4962158, 402.6548462, -933.1510620, 933.1510620
3: -205.3341675, 519.1251221, -205.3341675, 519.1251221, -724.4591675, 724.4591675
4: -590.9782715, 398.9001770, -590.9782715, 398.9001770, -989.8784180, 989.8782959

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0391165, upper bound: 808.8221868
time: 0.71 seconds

## Relational analysis of IS_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2033746, upper bound: 807.3290217
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1344858, upper bound: 806.2182831
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.40 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 4.40
Output dim: 4, lower bound: -805.8369742, upper bound: 806.2033746
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.40
Output dim: 4, lower bound: -805.8369742, upper bound: 807.7354474
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.40
Output dim: 4, lower bound: -806.2179647, upper bound: 807.6027596
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.40
Output dim: 4, lower bound: -806.2179647, upper bound: 807.9628608
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.40
Output dim: 4, lower bound: -807.7354458, upper bound: 805.8380044
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.40
Output dim: 4, lower bound: -808.1344830, upper bound: 806.2182824
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.40
Output dim: 4, lower bound: -806.2033746, upper bound: 807.3290217
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.40
Output dim: 4, lower bound: -808.1344858, upper bound: 806.2182831

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -251.5948334, 219.6874542, -422.4466553, 354.7306213, -606.3254395, 642.1340942
1: -200.1122742, 214.7951965, -338.4577942, 344.2090454, -544.3212280, 553.2528076
2: -288.3356018, 235.8116150, -491.3921204, 376.1318054, -664.4672852, 727.2037354
3: -120.3001709, 288.3111572, -192.2084351, 481.8464050, -602.1466064, 480.5195618
4: -322.8778687, 232.7944031, -547.5710449, 372.2809143, -695.1587524, 780.3654785

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.8074023, upper bound: 807.7353988
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8369320, upper bound: 806.2169859
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.8369320, upper bound: 805.8369320
time: 0.69 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -332.7971191, 281.7926941, -443.1754761, 371.6176147, -704.4147339, 724.9680786
1: -265.7110901, 274.8519287, -355.3739929, 360.4365540, -626.1476440, 630.2257080
2: -384.1240845, 300.9845886, -516.3179321, 393.4807129, -777.6046143, 817.3024902
3: -153.6954041, 379.7464600, -200.7738037, 506.0827637, -659.7781982, 580.5202637
4: -429.0441895, 297.0279236, -575.2116089, 389.6814880, -818.7256470, 872.2395020

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2179647, upper bound: 807.6027596
time: 0.75 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2179647, upper bound: 807.6027596
time: 0.85 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -339.1815186, 285.2177429, -443.3177490, 371.9735413, -711.1550293, 728.5355225
1: -270.9281616, 278.1785583, -355.4833374, 360.7223816, -631.6505127, 633.6618042
2: -391.5458069, 304.4880066, -516.4367676, 393.7322693, -785.2780762, 820.9246826
3: -155.5240631, 386.5900574, -200.9066010, 506.2773132, -661.8013916, 587.4964600
4: -436.9592896, 300.3887939, -575.3381958, 389.9696960, -826.9289551, 875.7269897

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9385067, upper bound: 807.8293769
time: 0.67 seconds

## Relational analysis of IS_A1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2179647, upper bound: 807.9628608
time: 0.68 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2179647, upper bound: 807.9628608
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -420.1654358, 353.6795654, -251.5948334, 219.6874542, -639.8529053, 605.2744141
1: -336.6407471, 342.8673096, -200.1122742, 214.7951965, -551.4359131, 542.9796143
2: -489.1600037, 374.4150391, -288.3356018, 235.8116150, -724.9714966, 662.7504883
3: -191.1345062, 479.4635925, -120.3001709, 288.3111572, -479.4456787, 599.7637329
4: -545.2463989, 370.9644775, -322.8778687, 232.7944031, -778.0407104, 693.8423462

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7353972, upper bound: 805.8372757
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2033732, upper bound: 805.8369742
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2033732, upper bound: 805.8380044
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -452.7537842, 378.6615601, -348.0386353, 292.4441223, -745.1978149, 726.7000732
1: -363.0274048, 367.1170654, -278.1151123, 285.1520081, -648.1794434, 645.2321167
2: -527.7424316, 400.6523743, -402.1842957, 312.0704346, -839.8126831, 802.8366089
3: -204.3902130, 516.4533081, -159.7416534, 396.7953796, -601.1856079, 676.1949463
4: -587.9271851, 396.8180847, -448.7807617, 307.8369446, -895.7640991, 845.5988770

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8293920, upper bound: 805.9386386
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6027576, upper bound: 806.2179647
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9628586, upper bound: 806.2179647
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -429.8060913, 359.7723999, -452.7537842, 378.6615601, -808.4675903, 812.5260010
1: -344.2817688, 349.0133667, -363.0274048, 367.1170654, -711.3988037, 712.0407715
2: -500.1039734, 381.3972168, -527.7424316, 400.6523743, -900.7563477, 909.1396484
3: -194.8598633, 489.6657715, -204.3902130, 516.4533081, -711.3131714, 694.0559082
4: -557.3035278, 377.4635315, -587.9271851, 396.8180847, -954.1215210, 965.3907471

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9952486, upper bound: 808.8010121
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9962168, upper bound: 808.9962432
time: 0.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.27 seconds
IS_A1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 4, lower bound: -805.8369320, upper bound: 806.2169859
IS_A1_A1_B2_B2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 4, lower bound: -805.8369320, upper bound: 805.8369320
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 4, lower bound: -806.2179647, upper bound: 807.6027596
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 4, lower bound: -806.2179647, upper bound: 807.6027596
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 4, lower bound: -806.2179647, upper bound: 807.9628608
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 4, lower bound: -806.2179647, upper bound: 807.9628608
IS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 4, lower bound: -806.2033732, upper bound: 805.8369742
IS_A2_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 4, lower bound: -806.2033732, upper bound: 805.8380044
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 4, lower bound: -807.6027576, upper bound: 806.2179647
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 4, lower bound: -807.9628586, upper bound: 806.2179647
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 4, lower bound: -808.9952486, upper bound: 808.8010121
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 4, lower bound: -808.9962168, upper bound: 808.9962432

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -332.7971191, 281.7926941, -431.8000793, 363.2170410, -696.0141602, 713.5927734
1: -265.7110901, 274.8519287, -345.9753723, 352.3334656, -618.0445557, 620.8272095
2: -384.1240845, 300.9845886, -502.5689392, 384.6166382, -768.7406616, 803.5535278
3: -153.6954041, 379.7464600, -196.0210724, 493.0811462, -646.7765503, 575.7675171
4: -429.0441895, 297.0279236, -560.2510986, 381.0561218, -810.1003418, 857.2790527

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_A1_B1_A1

### Relational analysis result of IS_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9772057, upper bound: 807.5472580
time: 0.68 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -803.6458767, upper bound: 806.5671139
time: 0.71 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2173228, upper bound: 806.2174251
time: 0.82 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2173228, upper bound: 807.6027596
time: 0.66 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -332.7971191, 281.7926941, -435.6227112, 365.6372070, -698.4343262, 717.4154053
1: -265.7110901, 274.8519287, -349.2161560, 354.6196289, -620.3306885, 624.0679321
2: -384.1240845, 300.9845886, -507.0864868, 387.1161804, -771.2400513, 808.0709229
3: -153.6954041, 379.7464600, -197.5857086, 497.3825989, -651.0780029, 577.3320923
4: -429.0441895, 297.0279236, -564.9703369, 383.4651184, -812.5092773, 861.9982910

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_A1_B2_A1

### Relational analysis result of IS_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9772057, upper bound: 807.5472580
time: 0.85 seconds

## Relational analysis of IS_A1_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -803.6458767, upper bound: 806.5671139
time: 0.70 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2173228, upper bound: 806.2174251
time: 0.64 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.2173228, upper bound: 807.6027596
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -339.1815186, 285.2177429, -431.7549744, 363.1815186, -702.3630371, 716.9727173
1: -270.9281616, 278.1785583, -345.9388733, 352.2986755, -623.2268066, 624.1174316
2: -391.5458069, 304.4880066, -502.5156555, 384.5777893, -776.1235962, 807.0036621
3: -155.5240631, 386.5900574, -196.0006714, 493.0300293, -648.5540771, 582.5905762
4: -436.9592896, 300.3887939, -560.1921997, 381.0189209, -817.9782104, 860.5809937

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_A2_B1_A1

### Relational analysis result of IS_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9772057, upper bound: 807.9231858
time: 0.70 seconds

## Relational analysis of IS_A1_A2_A2_B1_A2

### Relational analysis result of IS_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7843643, upper bound: 807.9325594
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -339.1815186, 285.2177429, -435.6227112, 365.6372070, -704.8187256, 720.8404541
1: -270.9281616, 278.1785583, -349.2161560, 354.6196289, -625.5477905, 627.3947144
2: -391.5458069, 304.4880066, -507.0864868, 387.1161804, -778.6618652, 811.5743408
3: -155.5240631, 386.5900574, -197.5857086, 497.3825989, -652.9066772, 584.1754761
4: -436.9592896, 300.3887939, -564.9703369, 383.4651184, -820.4244385, 865.3591309

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_A2_B2_A1

### Relational analysis result of IS_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9772057, upper bound: 807.9231858
time: 0.72 seconds

## Relational analysis of IS_A1_A2_A2_B2_A2

### Relational analysis result of IS_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7843643, upper bound: 807.9325594
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -449.6639404, 375.8992615, -332.7971191, 281.7926941, -731.4566650, 708.6964111
1: -360.5164185, 364.4870300, -265.7110901, 274.8519287, -635.3682861, 630.1981201
2: -524.0574951, 397.8850708, -384.1240845, 300.9845886, -825.0421143, 782.0090942
3: -202.9973450, 512.8676758, -153.6954041, 379.7464600, -582.7437744, 666.5631104
4: -583.8496094, 394.0246277, -429.0441895, 297.0279236, -880.8775635, 823.0687256

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6027576, upper bound: 806.2179647
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6027576, upper bound: 806.2179647
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -449.9850159, 376.3931580, -339.1815186, 285.2177429, -735.2026978, 715.5747070
1: -360.7696838, 364.9222412, -270.9281616, 278.1785583, -638.9482422, 635.8502808
2: -524.3867188, 398.2768250, -391.5458069, 304.4880066, -828.8746948, 789.8226318
3: -203.1929169, 513.2659912, -155.5240631, 386.5900574, -589.7827148, 668.7900391
4: -584.2110596, 394.4639893, -436.9592896, 300.3887939, -884.5998535, 831.4232788

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8293752, upper bound: 805.9385067
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2033548, upper bound: 806.2169864
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2033548, upper bound: 806.2179647
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -429.8060913, 359.7723999, -408.1961670, 344.1016541, -773.9076538, 767.9684448
1: -344.2817688, 349.0133667, -327.0793457, 333.3669434, -677.6486206, 676.0926514
2: -500.1039734, 381.3972168, -475.4765015, 363.8530273, -863.9570312, 856.8736572
3: -194.8598633, 489.6657715, -185.4267883, 465.6343689, -660.4942627, 675.0924072
4: -557.3035278, 377.4635315, -530.2349243, 360.8494568, -918.1529541, 907.6984863

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8663113, upper bound: 807.8641412
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9790193, upper bound: 808.8005999
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -412.6676941, 345.1517334, -706.3592529, 572.3082886, -981.6872559, 1051.5109863
1: -330.3198853, 334.7621155, -568.0347900, 555.8222046, -882.9141235, 902.7968750
2: -479.5230103, 365.9483032, -822.0452881, 605.1855469, -1081.6247559, 1187.9936523
3: -187.2959595, 469.5303345, -312.1067505, 795.0562744, -982.3522339, 778.3438721
4: -534.5791626, 362.3482361, -913.7318726, 598.2418823, -1130.0872803, 1276.0799561

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9954316, upper bound: 808.5518607
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9958396, upper bound: 808.9958443
time: 0.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.34 seconds
IS_A1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 4, lower bound: -806.2173228, upper bound: 806.2174251
IS_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 4, lower bound: -806.2173228, upper bound: 807.6027596
IS_A1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 4, lower bound: -806.2173228, upper bound: 806.2174251
IS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 4, lower bound: -806.2173228, upper bound: 807.6027596
IS_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 4, lower bound: -805.9772057, upper bound: 807.9231858
IS_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 4, lower bound: -805.7843643, upper bound: 807.9325594
IS_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 4, lower bound: -805.9772057, upper bound: 807.9231858
IS_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 4, lower bound: -805.7843643, upper bound: 807.9325594
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 4, lower bound: -807.6027576, upper bound: 806.2179647
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 4, lower bound: -807.6027576, upper bound: 806.2179647
IS_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 4, lower bound: -806.2033548, upper bound: 806.2169864
IS_A2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.34
Output dim: 4, lower bound: -806.2033548, upper bound: 806.2179647
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 4, lower bound: -808.8663113, upper bound: 807.8641412
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 4, lower bound: -808.9790193, upper bound: 808.8005999
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 4, lower bound: -808.9954316, upper bound: 808.5518607
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.34
Output dim: 4, lower bound: -808.9958396, upper bound: 808.9958443

## BFS IS instance: IS_A1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -332.7971191, 281.7926941, -435.2050171, 365.2911072, -698.0882568, 716.9976196
1: -265.7110901, 274.8519287, -348.6129150, 354.1904907, -619.9015503, 623.4647827
2: -384.1240845, 300.9845886, -506.7109680, 386.6527710, -770.7766724, 807.6954346
3: -153.6954041, 379.7464600, -197.0384827, 496.4768066, -650.1722412, 576.7849121
4: -429.0441895, 297.0279236, -564.9388428, 383.2071228, -812.2512817, 861.9667969

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.9805034, upper bound: 805.4649928
time: 0.76 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2170826, upper bound: 806.2170773
time: 0.71 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -332.7971191, 281.7926941, -442.5560303, 370.2713623, -703.0684814, 724.3487549
1: -265.7110901, 274.8519287, -354.7106934, 359.0053101, -624.7164307, 629.5626221
2: -384.1240845, 300.9845886, -515.3446045, 391.8888855, -776.0128784, 816.3292236
3: -153.6954041, 379.7464600, -199.9835968, 504.6768494, -658.3722534, 579.7300415
4: -429.0441895, 297.0279236, -574.1887817, 388.1746826, -817.2187500, 871.2166748

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2169835, upper bound: 806.0344092
time: 0.69 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2169851, upper bound: 807.1325781
time: 0.64 seconds

## BFS IS instance: IS_A1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -307.2550049, 260.7466431, -421.3722839, 355.3727417, -662.6277466, 682.1188965
1: -245.2618866, 254.3102875, -337.5812988, 344.6553345, -589.9170532, 591.8916016
2: -354.1973267, 278.2404785, -490.3989868, 376.1895752, -730.3866577, 768.6394043
3: -142.3887634, 350.6885681, -191.6089935, 481.4836121, -623.8723755, 542.2975464
4: -395.7063293, 274.7497559, -546.8214722, 372.8684387, -768.5747681, 821.5712280

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7840665, upper bound: 807.3084464
time: 0.82 seconds

## Relational analysis of IS_A1_A2_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7840665, upper bound: 807.3084464
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -409.5816040, 340.8766785, -428.3564148, 360.5387573, -770.1203613, 769.2330933
1: -327.7558289, 331.2249756, -343.1787415, 349.7093506, -677.4651489, 674.4036865
2: -473.8589478, 361.9745483, -498.4827576, 381.7423096, -855.6012573, 860.4572754
3: -187.0029297, 465.8259888, -194.5700836, 489.1656799, -676.1685181, 660.3959961
4: -528.5266724, 358.3767395, -555.7480469, 378.2663574, -906.7930298, 914.1247559

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B1_A2_A1

### Relational analysis result of IS_A1_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7564226, upper bound: 807.8676177
time: 0.65 seconds

## Relational analysis of IS_A1_A2_A2_B1_A2_A2

### Relational analysis result of IS_A1_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7840578, upper bound: 807.9413001
time: 0.64 seconds

## BFS IS instance: IS_A1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -307.2550049, 260.7466431, -425.0611267, 357.7640991, -665.0191040, 685.8076172
1: -245.2618866, 254.3102875, -340.7063293, 346.8792419, -592.1410522, 595.0166016
2: -354.1973267, 278.2404785, -494.7663574, 378.4607239, -732.6580200, 773.0068359
3: -142.3887634, 350.6885681, -193.1617279, 485.6674805, -628.0561523, 543.8502808
4: -395.7063293, 274.7497559, -551.3928833, 375.1279907, -770.8343506, 826.1426392

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.5562224, upper bound: 807.1165808
time: 0.63 seconds

## Relational analysis of IS_A1_A2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9771565, upper bound: 807.9231858
time: 0.66 seconds

## BFS IS instance: IS_A1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -409.5816040, 340.8766785, -432.0444641, 362.8468933, -772.4284668, 772.9211426
1: -327.7558289, 331.2249756, -346.3021851, 351.8652039, -679.6209717, 677.5271606
2: -473.8589478, 361.9745483, -502.8375549, 384.1100464, -857.9689941, 864.8120728
3: -187.0029297, 465.8259888, -196.0796509, 493.2920532, -680.2948608, 661.9055176
4: -528.5266724, 358.3767395, -560.3005371, 380.5333862, -909.0600586, 918.6772461

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.6209531, upper bound: 805.6445069
time: 0.70 seconds

## Relational analysis of IS_A1_A2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.6209533, upper bound: 805.8366266
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -435.4188843, 365.4100342, -332.7971191, 281.7926941, -717.2115479, 698.2070923
1: -348.7833862, 354.3109131, -265.7110901, 274.8519287, -623.6350708, 620.0219727
2: -506.9671631, 386.7787476, -384.1240845, 300.9845886, -807.9517822, 770.9027710
3: -197.0967712, 496.7078552, -153.6954041, 379.7464600, -576.8432007, 650.4032593
4: -565.2241821, 383.3228455, -429.0441895, 297.0279236, -862.2520752, 812.3668823

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5472560, upper bound: 805.9772057
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.5671130, upper bound: 803.6458767
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 20

Time for candidate selection: 8.92 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5784228, upper bound: 806.2177954
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.3951092, upper bound: 805.6228456
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5680234, upper bound: 805.2929108
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -803.7374135, upper bound: 805.2909212
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -442.5560303, 370.2713623, -332.7971191, 281.7926941, -724.3486328, 703.0684814
1: -354.7106934, 359.0053101, -265.7110901, 274.8519287, -629.5625610, 624.7164307
2: -515.3446045, 391.8888855, -384.1240845, 300.9845886, -816.3292236, 776.0128174
3: -199.9835968, 504.6768494, -153.6954041, 379.7464600, -579.7300415, 658.3722534
4: -574.1887817, 388.1746826, -429.0441895, 297.0279236, -871.2166748, 817.2188110

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5472560, upper bound: 805.9772057
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.5671130, upper bound: 803.6458767
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 20

Time for candidate selection: 9.09 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5784228, upper bound: 806.2177954
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.3951092, upper bound: 805.6228577
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5680234, upper bound: 805.2929108
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -803.7374135, upper bound: 805.2910598
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -404.9685364, 339.2808838, -354.2401428, 298.6145325, -703.5830688, 693.5209961
1: -324.1711731, 329.1825867, -283.7326050, 289.6389160, -613.8100586, 612.9151611
2: -470.6492615, 359.9148254, -411.7906799, 316.2800903, -786.9293213, 771.7054443
3: -183.7050629, 461.1755981, -160.7036591, 404.3106079, -588.0156860, 621.8792725
4: -524.5393677, 356.1525574, -458.6590881, 313.3552551, -837.8945923, 814.8116455

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8662798, upper bound: 807.8641412
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8662798, upper bound: 807.8641412
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -428.3950500, 358.6085510, -400.5152588, 337.3289185, -765.7239990, 759.1237183
1: -343.1346130, 347.8863525, -320.8334351, 326.8308411, -669.9653931, 668.7197266
2: -498.4352722, 380.1820068, -466.3867798, 356.7891235, -855.2243652, 846.5687256
3: -194.2423248, 488.0565186, -181.8879547, 456.7010498, -650.9433594, 669.9444580
4: -555.4625854, 376.2602844, -520.2033691, 353.8252258, -909.2877808, 896.4635620

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022590, upper bound: 808.5005852
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2021922, upper bound: 808.0851681
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -377.0955811, 318.7004395, -572.0731201, 471.3119812, -841.8922119, 890.7735596
1: -301.5436096, 309.1554565, -460.3016663, 457.8415222, -753.7570190, 769.4570923
2: -437.7516174, 338.3402710, -665.8199463, 499.1508789, -930.5166016, 1004.1602173
3: -172.9016418, 430.0166321, -257.5309143, 647.1233521, -820.0250244, 682.1652222
4: -488.2969971, 334.9786377, -739.8893433, 493.1993408, -975.0884399, 1074.8679199

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1763225, upper bound: 808.1756101
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9678608, upper bound: 806.7222028
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9120327, upper bound: 806.7222449
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -409.5754700, 342.6080627, -695.6989746, 562.1867676, -969.7088623, 1038.3068848
1: -327.8423157, 332.3361816, -559.4201660, 546.1665649, -871.6173706, 891.7563477
2: -475.8623047, 363.3382263, -809.3776855, 594.6675415, -1068.5819092, 1172.7159424
3: -185.9459686, 466.0455017, -306.5054626, 782.2507935, -968.1966553, 770.0744629
4: -530.5190430, 359.7381897, -899.7716064, 587.6741943, -1116.7974854, 1259.5096436

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7998045, upper bound: 808.8134866
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7998045, upper bound: 808.7986718
time: 0.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.13 seconds
IS_A1_A2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 4, lower bound: -805.9805034, upper bound: 805.4649928
IS_A1_A2_A1_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 4, lower bound: -806.2170826, upper bound: 806.2170773
IS_A1_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 4, lower bound: -806.2169835, upper bound: 806.0344092
IS_A1_A2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 4, lower bound: -806.2169851, upper bound: 807.1325781
IS_A1_A2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 4, lower bound: -805.7840665, upper bound: 807.3084464
IS_A1_A2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 4, lower bound: -805.7840665, upper bound: 807.3084464
IS_A1_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 4, lower bound: -805.7564226, upper bound: 807.8676177
IS_A1_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 4, lower bound: -805.7840578, upper bound: 807.9413001
IS_A1_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 4, lower bound: -805.5562224, upper bound: 807.1165808
IS_A1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 4, lower bound: -805.9771565, upper bound: 807.9231858
IS_A1_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 4, lower bound: -805.6209531, upper bound: 805.6445069
IS_A1_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 4, lower bound: -805.6209533, upper bound: 805.8366266
IS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 4, lower bound: -807.5680234, upper bound: 805.2929108
IS_A2_B1_B2_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 4, lower bound: -803.7374135, upper bound: 805.2909212
IS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 4, lower bound: -807.5680234, upper bound: 805.2929108
IS_A2_B1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 4, lower bound: -803.7374135, upper bound: 805.2910598
IS_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 4, lower bound: -808.8662798, upper bound: 807.8641412
IS_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 4, lower bound: -808.8662798, upper bound: 807.8641412
IS_A2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 4, lower bound: -808.2022590, upper bound: 808.5005852
IS_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 4, lower bound: -808.2021922, upper bound: 808.0851681
IS_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 4, lower bound: -808.9678608, upper bound: 806.7222028
IS_A2_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 4, lower bound: -806.9120327, upper bound: 806.7222449
IS_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 4, lower bound: -808.7998045, upper bound: 808.8134866
IS_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 4, lower bound: -808.7998045, upper bound: 808.7986718

## BFS IS instance: IS_A1_A2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -276.6354675, 240.0173035, -392.7685852, 334.3920593, -611.0275269, 632.7858887
1: -220.7916718, 234.1406403, -314.3935547, 324.3073120, -545.0989380, 548.5339966
2: -318.9915771, 256.6457825, -456.6623230, 354.3316040, -673.3231812, 713.3078613
3: -131.3087158, 318.5422974, -180.1191864, 449.5715027, -580.8801270, 498.6614685
4: -356.4848022, 253.5630951, -509.3583679, 351.1623535, -707.6470337, 762.9213867

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7563022, upper bound: 807.2610004
time: 0.72 seconds

## Relational analysis of IS_A1_A2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7563023, upper bound: 807.3373124
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -401.0169983, 333.6079102, -425.2744751, 358.0187988, -759.0357666, 758.8823242
1: -320.5706177, 324.1984558, -340.7044983, 347.3022156, -667.8728027, 664.9029541
2: -463.0681458, 354.3021851, -494.8314819, 379.1893921, -842.2575684, 849.1336670
3: -183.1567535, 455.7034302, -193.2539978, 485.7111511, -668.8679199, 648.9573364
4: -516.6904297, 350.8312988, -551.7014160, 375.6733093, -892.3637695, 902.5327148

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7839308, upper bound: 807.3082202
time: 0.72 seconds

## Relational analysis of IS_A1_A2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7839308, upper bound: 807.3227809
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -305.6888428, 259.4651184, -417.8401184, 351.4624634, -657.1513062, 677.3052368
1: -243.9965515, 253.0748901, -334.8401184, 340.7930603, -584.7895508, 587.9149170
2: -352.3628235, 276.8963623, -486.2316284, 371.9615479, -724.3243408, 763.1278687
3: -141.7517853, 348.9402466, -189.8629456, 477.4236755, -619.1754761, 538.8032227
4: -393.6826172, 273.4249268, -541.9742432, 368.5931702, -762.2757568, 815.3991089

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.9767559, upper bound: 806.8397343
time: 0.61 seconds

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.9767559, upper bound: 807.9231858
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -412.7009277, 347.3033752, -332.7971191, 281.7926941, -694.4935303, 680.1004639
1: -330.3798218, 336.9181824, -265.7110901, 274.8519287, -605.2316895, 602.6292725
2: -480.3560791, 367.9341431, -384.1240845, 300.9845886, -781.3406982, 752.0579834
3: -187.2425537, 471.3845825, -153.6954041, 379.7464600, -566.9890137, 625.0799561
4: -535.8590698, 364.6180725, -429.0441895, 297.0279236, -832.8869629, 793.6621704

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.0258918, upper bound: 802.9861194
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7785968, upper bound: 805.2928395
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9454582, upper bound: 804.4764904
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9408541, upper bound: 805.2449191
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1022184, upper bound: 805.2449028
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -420.7076416, 352.7497864, -332.7971191, 281.7926941, -702.5002441, 685.5468750
1: -336.9926453, 342.1614685, -265.7110901, 274.8519287, -611.8444214, 607.8725586
2: -489.6965332, 373.5150452, -384.1240845, 300.9845886, -790.6810303, 757.6391602
3: -190.3265991, 480.1835022, -153.6954041, 379.7464600, -570.0730591, 633.8789062
4: -545.8570557, 370.0228577, -429.0441895, 297.0279236, -842.8850098, 799.0670166

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5226406, upper bound: 805.2927019
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3330774, upper bound: 804.4759404
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.9687066, upper bound: 803.6500996
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3408673, upper bound: 805.2442232
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5679868, upper bound: 805.2444491
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -363.7673645, 304.7037354, -354.2401428, 298.6145325, -662.3818970, 658.9438477
1: -291.1853027, 295.7985535, -283.7326050, 289.6389160, -580.8242188, 579.5311279
2: -422.4324646, 323.4919128, -411.7906799, 316.2800903, -738.7125244, 735.2825928
3: -164.8517761, 414.1521912, -160.7036591, 404.3106079, -569.1622925, 574.8558350
4: -470.2705994, 319.9736328, -458.6590881, 313.3552551, -783.6257935, 778.6326904

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8635242, upper bound: 807.8626466
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8635241, upper bound: 807.8626466
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -422.2749634, 353.2666016, -354.2401428, 298.6145325, -720.8895264, 707.5067139
1: -338.1520996, 342.7183228, -283.7326050, 289.6389160, -627.7910156, 626.4509277
2: -491.1721497, 374.6179810, -411.7906799, 316.2800903, -807.4522705, 786.4085693
3: -191.4285736, 480.9315491, -160.7036591, 404.3106079, -595.7391968, 641.6351318
4: -547.4479370, 370.7572632, -458.6590881, 313.3552551, -860.8031006, 829.4163208

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8635241, upper bound: 807.8636044
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.8635241, upper bound: 807.8641412
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -425.5954285, 356.1755981, -383.9626465, 324.9349365, -750.5303955, 740.1382446
1: -340.8552856, 345.5572815, -307.2273560, 314.9211731, -655.7763672, 652.7845459
2: -495.0787354, 377.6608276, -446.4984741, 343.9260254, -839.0047607, 824.1591797
3: -192.9711914, 484.7954102, -175.0340271, 437.7497864, -630.7209473, 659.8293457
4: -551.7547607, 373.7503662, -498.4967957, 341.1781921, -892.9328613, 872.2471313

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2021922, upper bound: 808.0851681
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2021922, upper bound: 808.0851681
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -425.8152466, 356.4920654, -390.4650574, 329.1775818, -754.9927979, 746.9571533
1: -341.0280762, 345.8348999, -312.6345520, 318.9571228, -659.9851685, 658.4694214
2: -495.2949219, 377.9476318, -454.1794434, 348.1879578, -843.4827881, 832.1270142
3: -193.1141510, 485.0729980, -177.6081848, 445.1135254, -638.2276611, 662.6811523
4: -551.9886475, 374.0677795, -506.6883545, 345.3511658, -897.3398438, 880.7561035

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2021925, upper bound: 808.0851681
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2021922, upper bound: 808.0851681
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -375.1574707, 317.2591248, -572.0675659, 471.3073425, -839.9171143, 889.3265381
1: -299.9717102, 307.7640381, -460.2972107, 457.8369751, -752.1546631, 768.0612793
2: -435.4510193, 336.8258667, -665.8135376, 499.1459351, -928.1818848, 1002.6394043
3: -172.1346893, 427.8754883, -257.5284424, 647.1171265, -819.2518311, 680.0260620
4: -485.7616882, 333.4869995, -739.8821411, 493.1946106, -972.5191040, 1073.3691406

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5257669, upper bound: 806.7217609
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5257669, upper bound: 805.1328913
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -380.1874390, 320.5768433, -691.8547974, 559.1325073, -937.4171753, 1012.4316406
1: -304.2405396, 310.7821045, -556.3065186, 543.1970825, -845.1865234, 867.0886230
2: -441.9090576, 339.6157227, -804.8543701, 591.4642944, -1031.4262695, 1144.4698486
3: -173.6408691, 433.1943054, -304.8825684, 778.0196533, -951.6605225, 735.4884033
4: -493.1286316, 336.5455322, -894.7669067, 584.5429688, -1076.2891846, 1231.3121338

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7994975, upper bound: 808.1764399
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0766353, upper bound: 808.0879441
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -681.6501465, 551.8124390, -696.3657837, 562.7486572, -1236.1340332, 1238.7591553
1: -548.0568848, 536.1000366, -559.9593506, 546.7133789, -1087.9565430, 1088.3455811
2: -792.7862549, 583.8844604, -810.1624756, 595.2567749, -1380.4807129, 1385.2854004
3: -301.1659241, 766.7492676, -306.8024292, 782.9974976, -1079.0104980, 1068.9896240
4: -881.3311768, 576.9235229, -900.6437378, 588.2474365, -1461.5876465, 1468.3554688

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9817874, upper bound: 808.0309628
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9817874, upper bound: 808.7983016
time: 0.87 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.00 seconds
IS_A1_A2_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -805.7563022, upper bound: 807.2610004
IS_A1_A2_A2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -805.7563023, upper bound: 807.3373124
IS_A1_A2_A2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -805.7839308, upper bound: 807.3082202
IS_A1_A2_A2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -805.7839308, upper bound: 807.3227809
IS_A1_A2_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -805.9767559, upper bound: 806.8397343
IS_A1_A2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -805.9767559, upper bound: 807.9231858
IS_A2_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -807.9408541, upper bound: 805.2449191
IS_A2_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -808.1022184, upper bound: 805.2449028
IS_A2_B1_B2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 4, lower bound: -807.3408673, upper bound: 805.2442232
IS_A2_B1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -807.5679868, upper bound: 805.2444491
IS_A2_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -807.8635242, upper bound: 807.8626466
IS_A2_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -807.8635241, upper bound: 807.8626466
IS_A2_B2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -807.8635241, upper bound: 807.8636044
IS_A2_B2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -807.8635241, upper bound: 807.8641412
IS_A2_B2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -808.2021922, upper bound: 808.0851681
IS_A2_B2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -808.2021922, upper bound: 808.0851681
IS_A2_B2_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -808.2021925, upper bound: 808.0851681
IS_A2_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -808.2021922, upper bound: 808.0851681
IS_A2_B2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -808.5257669, upper bound: 806.7217609
IS_A2_B2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -808.5257669, upper bound: 805.1328913
IS_A2_B2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -808.7994975, upper bound: 808.1764399
IS_A2_B2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -808.0766353, upper bound: 808.0879441
IS_A2_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -807.9817874, upper bound: 808.0309628
IS_A2_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 4, lower bound: -807.9817874, upper bound: 808.7983016

## BFS IS instance: IS_A1_A2_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -305.6888428, 259.4651184, -504.4114075, 418.9852295, -724.6740723, 763.8765259
1: -243.9965515, 253.0748901, -404.7279053, 406.1260071, -650.1223755, 657.8027344
2: -352.3628235, 276.8963623, -587.7135620, 443.1398315, -795.5026245, 864.6098633
3: -141.7517853, 348.9402466, -227.6819611, 574.5734863, -716.3252563, 576.6221924
4: -393.6826172, 273.4249268, -654.6939087, 439.6098022, -833.2924194, 928.1188354

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.7828304, upper bound: 805.7831628
time: 0.63 seconds

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7828290, upper bound: 807.9231858
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -404.0533752, 340.5291748, -332.7971191, 281.7926941, -685.8459473, 673.3262939
1: -323.3573914, 330.1812744, -265.7110901, 274.8519287, -598.2092896, 595.8923340
2: -470.1757202, 360.5476379, -384.1240845, 300.9845886, -771.1602173, 744.6716309
3: -183.4913635, 461.5953369, -153.6954041, 379.7464600, -563.2377930, 615.2907715
4: -524.6213379, 357.7170715, -429.0441895, 297.0279236, -821.6492920, 786.7611084

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7553080, upper bound: 805.2443209
time: 1.23 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9408541, upper bound: 805.2449191
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9064076, upper bound: 804.4741138
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9408541, upper bound: 805.2449191
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9408541, upper bound: 805.2449191
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -408.4045410, 344.0163879, -332.7971191, 281.7926941, -690.1970825, 676.8134766
1: -326.9043579, 333.6767883, -265.7110901, 274.8519287, -601.7562256, 599.3878784
2: -475.2729187, 364.3194580, -384.1240845, 300.9845886, -776.2575073, 748.4434204
3: -185.3458862, 466.6108704, -153.6954041, 379.7464600, -565.0923462, 620.3062744
4: -530.2327881, 361.0561829, -429.0441895, 297.0279236, -827.2607422, 790.1002808

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.0111162, upper bound: 802.9687290
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6969784, upper bound: 805.2442866
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9454582, upper bound: 804.4740828
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1022184, upper bound: 805.2449028
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -803.7373267, upper bound: 805.2425321
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -416.6333618, 349.5036316, -332.7971191, 281.7926941, -698.4258423, 682.3007202
1: -333.6996765, 339.0247498, -265.7110901, 274.8519287, -608.5515747, 604.7358398
2: -484.8765564, 370.0697021, -384.1240845, 300.9845886, -785.8610840, 754.1937256
3: -188.5165863, 475.6487122, -153.6954041, 379.7464600, -568.2630615, 629.3441162
4: -540.5172729, 366.6267700, -429.0441895, 297.0279236, -837.5451660, 795.6707153

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5226032, upper bound: 805.2441772
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.3330774, upper bound: 804.4734816
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.8670205, upper bound: 803.6482391
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.5679868, upper bound: 805.2444491
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.8801314, upper bound: 805.2426414
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -331.5585632, 280.0069580, -354.2401428, 298.6145325, -630.1730957, 634.2470703
1: -265.2190857, 271.8355408, -283.7326050, 289.6389160, -554.8580322, 555.5681152
2: -384.5210876, 297.1768494, -411.7906799, 316.2800903, -700.8011475, 708.9675293
3: -151.2190399, 377.8385620, -160.7036591, 404.3106079, -555.5296631, 538.5422363
4: -428.5667725, 294.1880493, -458.6590881, 313.3552551, -741.9219360, 752.8471069

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6382111, upper bound: 806.2222729
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6382155, upper bound: 807.8626466
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -578.5350342, 463.2528687, -354.2401428, 298.6145325, -877.1495361, 813.2226562
1: -464.9569702, 449.8235168, -283.7326050, 289.6389160, -754.5958862, 729.7203979
2: -671.9920654, 490.8324280, -411.7906799, 316.2800903, -988.2721558, 898.5853882
3: -254.2596588, 647.1934814, -160.7036591, 404.3106079, -654.6819458, 807.8971558
4: -746.5787964, 485.2071533, -458.6590881, 313.3552551, -1059.9340820, 940.0044556

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6382118, upper bound: 806.2226982
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.6382149, upper bound: 807.8643822
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -377.5877686, 318.1267700, -354.2401428, 298.6145325, -676.2022705, 672.3669434
1: -302.0976562, 308.5556946, -283.7326050, 289.6389160, -591.7365723, 592.2883301
2: -438.8045959, 337.3096313, -411.7906799, 316.2800903, -755.0847168, 749.1002808
3: -172.4472961, 429.9995728, -160.7036591, 404.3106079, -576.7579346, 590.7032471
4: -489.7282410, 334.0813293, -458.6590881, 313.3552551, -803.0834961, 792.7403564

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 16

Time for candidate selection: 7.36 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6390921, upper bound: 807.5207870
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6391662, upper bound: 807.6688925
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -637.8558350, 520.3029175, -354.2401428, 298.6145325, -936.4703369, 868.8936157
1: -512.5637207, 505.4349365, -283.7326050, 289.6389160, -802.2026367, 784.0610962
2: -741.4612427, 550.8605347, -411.7906799, 316.2800903, -1057.7413330, 957.0759888
3: -284.0897217, 718.7723389, -160.7036591, 404.3106079, -683.8062744, 879.4760132
4: -824.6390991, 544.0968018, -458.6590881, 313.3552551, -1137.9943848, 997.2734375

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 43

Time for candidate selection: 7.60 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2985847, upper bound: 807.6691214
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6391661, upper bound: 807.6694709
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -410.0061035, 344.7039795, -383.9626465, 324.9349365, -734.9410400, 728.6666260
1: -328.0672302, 334.4484558, -307.2273560, 314.9211731, -642.9883423, 641.6757202
2: -476.4688721, 365.6720276, -446.4984741, 343.9260254, -820.3948975, 812.1704712
3: -186.6450043, 467.1828003, -175.0340271, 437.7497864, -624.3947754, 642.2167358
4: -531.4525757, 362.0969543, -498.4967957, 341.1781921, -872.6307373, 860.5936890

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1881559, upper bound: 808.1513882
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 20

Time for candidate selection: 7.69 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2018713, upper bound: 808.1691914
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2018713, upper bound: 808.1690438
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -418.1563110, 350.1917114, -383.9626465, 324.9349365, -743.0912476, 734.1542358
1: -334.7731018, 339.7275085, -307.2273560, 314.9211731, -649.6942749, 646.9548340
2: -485.9559631, 371.2951355, -446.4984741, 343.9260254, -829.8819580, 817.7935791
3: -189.7548065, 476.2016907, -175.0340271, 437.7497864, -627.5045166, 651.2357178
4: -541.6577148, 367.5392761, -498.4967957, 341.1781921, -882.8359375, 866.0360718

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1881561, upper bound: 808.1513882
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 30

Time for candidate selection: 7.73 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2018713, upper bound: 808.1691914
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2018713, upper bound: 808.1690438
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -410.0061035, 344.7039795, -390.4650574, 329.1775818, -739.1836548, 735.1690674
1: -328.0672302, 334.4484558, -312.6345520, 318.9571228, -647.0242310, 647.0830078
2: -476.4688721, 365.6720276, -454.1794434, 348.1879578, -824.6568604, 819.8514404
3: -186.6450043, 467.1828003, -177.6081848, 445.1135254, -631.7585449, 644.7908936
4: -531.4525757, 362.0969543, -506.6883545, 345.3511658, -876.8037109, 868.7852173

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0851357, upper bound: 808.0849891
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0851357, upper bound: 808.0851681
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -418.1563110, 350.1917114, -390.4650574, 329.1775818, -747.3338623, 740.6567383
1: -334.7731018, 339.7275085, -312.6345520, 318.9571228, -653.7302246, 652.3620605
2: -485.9559631, 371.2951355, -454.1794434, 348.1879578, -834.1439209, 825.4746094
3: -189.7548065, 476.2016907, -177.6081848, 445.1135254, -634.8683472, 653.8098145
4: -541.6577148, 367.5392761, -506.6883545, 345.3511658, -887.0089111, 874.2276611

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1019604, upper bound: 806.8394270
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1883470, upper bound: 808.0652923
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -281.0052185, 246.5772858, -571.9486084, 471.2062378, -744.9951172, 818.5258789
1: -224.2363892, 239.3523102, -460.2008972, 457.7387085, -675.8088989, 699.5531006
2: -326.0074768, 262.4650269, -665.6733398, 499.0396118, -817.8747559, 928.1383057
3: -133.4230194, 324.3261108, -257.4750977, 646.9832153, -780.4061890, 576.2645264
4: -364.1661682, 259.8967590, -739.7265015, 493.0915222, -849.9423218, 999.6232910

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9549499, upper bound: 805.5250767
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9549499, upper bound: 805.1323762
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -398.0772400, 333.3607483, -571.8259277, 471.1020813, -863.5912476, 905.1866455
1: -318.6085815, 323.5118713, -460.1014709, 457.6373291, -771.2849731, 783.6133423
2: -462.2374268, 353.8119812, -665.5287476, 498.9301758, -955.7563477, 1019.3406982
3: -181.0343781, 453.1769104, -257.4201355, 646.8450317, -827.8793945, 705.5129395
4: -515.4379272, 350.2313232, -739.5658569, 492.9852905, -1003.1134033, 1089.7971191

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5257183, upper bound: 806.7222028
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5257183, upper bound: 806.7222028
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -363.4929504, 308.1613464, -688.1416626, 556.1199951, -916.8973389, 996.3028564
1: -290.5656738, 298.8183594, -553.2924194, 540.3136597, -827.9974976, 852.1107788
2: -421.9073486, 326.6832886, -800.4346924, 588.3245850, -1007.3520508, 1127.1179199
3: -166.8062286, 414.0898743, -303.3390198, 773.8367920, -940.6430054, 714.4094238
4: -471.2630920, 323.8132019, -889.9086304, 581.3973389, -1050.2413330, 1213.7218018

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1538843, upper bound: 808.1680898
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6778726, upper bound: 808.1762559
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -371.2197266, 313.3545837, -688.2384033, 556.3732910, -925.2749634, 1001.5929565
1: -296.9222412, 303.8417053, -553.3706055, 540.5280151, -834.8555908, 857.2122192
2: -430.9479675, 332.0665894, -800.5298462, 588.5676880, -1017.0692139, 1132.5964355
3: -169.8011017, 422.8096313, -303.4059753, 773.9666138, -943.7677002, 723.4866943
4: -481.0086365, 329.0751038, -890.0019531, 581.6806030, -1060.7681885, 1219.0766602

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0765222, upper bound: 808.0879403
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0765222, upper bound: 808.0879403
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -549.9054565, 452.8660889, -696.3657837, 562.7486572, -1102.2432861, 1135.6726074
1: -442.4114380, 440.0763245, -559.9593506, 546.7133789, -980.7439575, 989.2699585
2: -639.6183472, 479.8450928, -810.1624756, 595.2567749, -1225.2071533, 1277.1199951
3: -247.7129974, 621.8483276, -306.8024292, 782.9974976, -1023.0081177, 922.7175903
4: -710.8350830, 474.0637207, -900.6437378, 588.2474365, -1288.6203613, 1360.8016357

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0963976, upper bound: 808.1742558
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0970151, upper bound: 808.4028679
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -673.8658447, 544.3862305, -696.3657837, 562.7486572, -1228.2216797, 1232.0477295
1: -541.7666626, 529.0551147, -559.9593506, 546.7133789, -1081.5327148, 1081.7784424
2: -783.5416260, 576.1945190, -810.1624756, 595.2567749, -1371.0576172, 1378.2171631
3: -297.2240601, 757.4203491, -306.8024292, 782.9974976, -1075.4609375, 1059.5589600
4: -871.1616211, 569.1795654, -900.6437378, 588.2474365, -1451.2514648, 1461.3525391

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0963976, upper bound: 808.9760072
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1763225, upper bound: 808.1756140
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1763162, upper bound: 808.1756140
time: 0.72 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.50 seconds
IS_A1_A2_A2_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.50
Output dim: 4, lower bound: -805.7828304, upper bound: 805.7831628
IS_A1_A2_A2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -805.7828290, upper bound: 807.9231858
IS_A2_B1_B2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -807.9408541, upper bound: 805.2449191
IS_A2_B1_B2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -807.9408541, upper bound: 805.2449191
IS_A2_B1_B2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.1022184, upper bound: 805.2449028
IS_A2_B1_B2_B1_A1_A1_A2_A2, status: Status.VERIFIED, split count: 8, time: 5.50
Output dim: 4, lower bound: -803.7373267, upper bound: 805.2425321
IS_A2_B1_B2_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -807.5679868, upper bound: 805.2444491
IS_A2_B1_B2_B1_A2_A1_A2_A2, status: Status.VERIFIED, split count: 8, time: 5.50
Output dim: 4, lower bound: -804.8801314, upper bound: 805.2426414
IS_A2_B2_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -807.6382111, upper bound: 806.2222729
IS_A2_B2_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -807.6382155, upper bound: 807.8626466
IS_A2_B2_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -807.6382118, upper bound: 806.2226982
IS_A2_B2_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -807.6382149, upper bound: 807.8643822
IS_A2_B2_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.6390921, upper bound: 807.5207870
IS_A2_B2_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.6391662, upper bound: 807.6688925
IS_A2_B2_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.2985847, upper bound: 807.6691214
IS_A2_B2_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.6391661, upper bound: 807.6694709
IS_A2_B2_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.2018713, upper bound: 808.1691914
IS_A2_B2_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.2018713, upper bound: 808.1690438
IS_A2_B2_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.2018713, upper bound: 808.1691914
IS_A2_B2_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.2018713, upper bound: 808.1690438
IS_A2_B2_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.0851357, upper bound: 808.0849891
IS_A2_B2_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.0851357, upper bound: 808.0851681
IS_A2_B2_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.1019604, upper bound: 806.8394270
IS_A2_B2_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.1883470, upper bound: 808.0652923
IS_A2_B2_A2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -807.9549499, upper bound: 805.5250767
IS_A2_B2_A2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -807.9549499, upper bound: 805.1323762
IS_A2_B2_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.5257183, upper bound: 806.7222028
IS_A2_B2_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.5257183, upper bound: 806.7222028
IS_A2_B2_A2_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.1538843, upper bound: 808.1680898
IS_A2_B2_A2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.6778726, upper bound: 808.1762559
IS_A2_B2_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.0765222, upper bound: 808.0879403
IS_A2_B2_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.0765222, upper bound: 808.0879403
IS_A2_B2_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.0963976, upper bound: 808.1742558
IS_A2_B2_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.0970151, upper bound: 808.4028679
IS_A2_B2_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.1763225, upper bound: 808.1756140
IS_A2_B2_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.50
Output dim: 4, lower bound: -808.1763162, upper bound: 808.1756140

## BFS IS instance: IS_A1_A2_A2_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -305.6888428, 259.4651184, -505.7521362, 419.1348267, -724.8236694, 765.2172241
1: -243.9965515, 253.0748901, -405.7545166, 406.3147583, -650.3111572, 658.8292236
2: -352.3628235, 276.8963623, -589.2426147, 443.3652344, -795.7279053, 866.1389160
3: -141.7517853, 348.9402466, -227.7870789, 575.4099121, -717.1616821, 576.7272339
4: -393.6826172, 273.4249268, -656.3441772, 439.8095398, -833.4920044, 929.7691040

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.6513669, upper bound: 807.7353154
time: 0.75 seconds

## Relational analysis of IS_A1_A2_A2_B2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.7826059, upper bound: 807.9231856
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -404.0533752, 340.5291748, -321.0038757, 272.8603210, -676.9136963, 661.5330811
1: -323.3573914, 330.1812744, -256.1362305, 266.2259521, -589.5833740, 586.3173218
2: -470.1757202, 360.5476379, -370.3713684, 291.5274658, -761.7030029, 730.9190063
3: -183.4913635, 461.5953369, -148.5743103, 366.7581482, -550.2495117, 610.1696777
4: -524.6213379, 357.7170715, -413.8730774, 287.7049255, -812.3262329, 771.5900879

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 18

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 10

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 18

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 10

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=1011.34521484375
rel_dist={4: [-809.0065995903992, 809.0065995903992]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.4327210, upper bound: 806.4685932
time: 0.62 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0035339, upper bound: 809.0035334
time: 0.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.54 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 4, lower bound: -807.4327210, upper bound: 806.4685932
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 4, lower bound: -809.0035339, upper bound: 809.0035334

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -435.6275024, 366.9198914, -355.7917175, 298.0696411, -733.6970825, 722.7116089
1: -349.4053650, 355.7280579, -284.4114380, 290.6557922, -640.0611572, 640.1395264
2: -507.5793762, 388.1117249, -411.3229675, 317.9904480, -825.5695801, 799.4346924
3: -197.9787903, 498.2431030, -162.9068451, 405.6134033, -603.5921021, 661.1499634
4: -565.4621582, 384.6995239, -458.9186401, 313.6753845, -879.1375732, 843.6181030

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7205285, upper bound: 805.8241682
time: 0.66 seconds

## Relational analysis of IS_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8669917, upper bound: 805.8373094
time: 0.69 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.2572202, upper bound: 806.2178128
time: 0.71 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -458.9994507, 383.7575378, -455.0569763, 380.6590271, -839.6583252, 838.8145142
1: -368.0681763, 372.0833130, -364.9006348, 369.0178833, -737.0860596, 736.9839478
2: -535.0563354, 405.9872131, -530.4962158, 402.6548462, -937.7111816, 936.4833984
3: -207.0673218, 523.5335693, -205.3341675, 519.1251221, -726.1923828, 728.8677368
4: -596.0374756, 402.1069641, -590.9782715, 398.9001770, -994.9375610, 993.0852051

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5041221, upper bound: 808.4239731
time: 0.68 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0030538, upper bound: 809.0030567
time: 0.82 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.32 seconds
IS_B1_B1, status: Status.VERIFIED, split count: 2, time: 3.32
Output dim: 4, lower bound: -806.8669917, upper bound: 805.8373094
IS_B1_B2, status: Status.VERIFIED, split count: 2, time: 3.32
Output dim: 4, lower bound: -807.2572202, upper bound: 806.2178128
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 4, lower bound: -805.5041221, upper bound: 808.4239731
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 4, lower bound: -809.0030538, upper bound: 809.0030567

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -459.1759033, 385.6335144, -455.0367737, 380.6445007, -839.8204346, 840.6702881
1: -368.0805664, 373.7598572, -364.8841553, 369.0036316, -737.0841064, 738.6440430
2: -535.2346802, 407.8876953, -530.4725342, 402.6394043, -937.8740845, 938.3601685
3: -207.7972870, 524.4395752, -205.3260040, 519.1027222, -726.8999023, 729.7655640
4: -596.5790405, 403.9531555, -590.9523926, 398.8849792, -995.4639893, 994.9055176

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.5865752, upper bound: 805.8864897
time: 0.63 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.5041195, upper bound: 808.4054583
time: 0.63 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -458.9501648, 383.7214966, -455.0569763, 380.6590271, -839.6091919, 838.7784424
1: -368.0279236, 372.0481262, -364.9006348, 369.0178833, -737.0457764, 736.9487305
2: -534.9984741, 405.9482117, -530.4962158, 402.6548462, -937.6533203, 936.4443970
3: -207.0472565, 523.4794312, -205.3341675, 519.1251221, -726.1723633, 728.8135376
4: -595.9743042, 402.0695801, -590.9782715, 398.9001770, -994.8743896, 993.0477295

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9974313, upper bound: 808.8210989
time: 0.68 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0029249, upper bound: 809.0029277
time: 0.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.10 seconds
IS_B2_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.10
Output dim: 4, lower bound: -804.5865752, upper bound: 805.8864897
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 4, lower bound: -805.5041195, upper bound: 808.4054583
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 4, lower bound: -808.9974313, upper bound: 808.8210989
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.10
Output dim: 4, lower bound: -809.0029249, upper bound: 809.0029277

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -459.0090332, 385.4901733, -451.1232605, 377.2937317, -836.3027344, 836.6134033
1: -367.9445190, 373.6210938, -361.6944580, 365.7542114, -733.6987305, 735.3155518
2: -535.0336914, 407.7363281, -525.7642822, 399.0956421, -934.1293335, 933.5006104
3: -207.7215729, 524.2442017, -203.5501556, 514.5461426, -722.2676392, 727.7943115
4: -596.3568115, 403.8040771, -585.7463379, 395.3947449, -991.7515869, 989.5502930

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -803.6086523, upper bound: 806.9284750
time: 0.64 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -803.6086523, upper bound: 808.4054584
time: 0.66 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -437.4374695, 366.9833679, -410.5120544, 346.1290894, -783.5665283, 777.4954224
1: -350.6418457, 355.7243347, -328.9624634, 335.2947083, -685.9365234, 684.6867676
2: -509.7126770, 388.1612244, -478.2442322, 365.9327087, -875.6453857, 866.4053955
3: -197.8142548, 498.9472046, -186.3758240, 468.3313599, -666.1453857, 685.3229980
4: -568.0895386, 384.6574402, -533.3006592, 362.9630737, -931.0526123, 917.9578857

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8205620, upper bound: 808.8205641
time: 0.77 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.8205620, upper bound: 808.8210989
time: 0.64 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -430.7584229, 359.3783875, -708.8129272, 574.3351440, -1001.7616577, 1068.1912842
1: -345.0603943, 348.5012512, -570.0178833, 557.7753906, -899.6048584, 918.5191650
2: -501.1179810, 380.6953735, -824.9526978, 607.2996216, -1105.2259521, 1205.6478271
3: -194.7231598, 490.1684875, -313.1855469, 797.8487549, -992.5718994, 799.9904175
4: -558.4459229, 377.0600891, -916.9576416, 600.3475342, -1155.9232178, 1294.0177002

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9900144, upper bound: 808.9844013
time: 0.76 seconds

## Relational analysis of IS_B2_A2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9898569, upper bound: 808.9898602
time: 0.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.75 seconds
IS_B2_A1_B2_A1, status: Status.VERIFIED, split count: 4, time: 3.75
Output dim: 4, lower bound: -803.6086523, upper bound: 806.9284750
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -803.6086523, upper bound: 808.4054584
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -808.8205620, upper bound: 808.8205641
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -808.8205620, upper bound: 808.8210989
IS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -808.9900144, upper bound: 808.9844013
IS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 4, lower bound: -808.9898569, upper bound: 808.9898602

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -455.2508545, 382.2695007, -451.1232605, 377.2937317, -832.5445557, 833.3927612
1: -364.8853149, 370.5006104, -361.6944580, 365.7542114, -730.6395264, 732.1950684
2: -530.5150757, 404.3332825, -525.7642822, 399.0956421, -929.6105957, 930.0974731
3: -206.0198669, 519.8605957, -203.5501556, 514.5461426, -720.5660400, 723.4107056
4: -591.3572388, 400.4524231, -585.7463379, 395.3947449, -986.7519531, 986.1987305

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -803.6075966, upper bound: 803.6075944
time: 0.62 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -803.6075966, upper bound: 807.7369912
time: 0.75 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -413.8527527, 348.7024841, -410.5120544, 346.1290894, -759.9818115, 759.2144165
1: -331.6310730, 337.8425293, -328.9624634, 335.2947083, -666.9257812, 666.8049927
2: -482.0812073, 368.7154236, -478.2442322, 365.9327087, -848.0138550, 846.9596558
3: -187.8464050, 472.0462036, -186.3758240, 468.3313599, -656.1774902, 658.4219971
4: -537.5671387, 365.6387634, -533.3006592, 362.9630737, -900.5302124, 898.9393921

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2218297, upper bound: 806.8983947
time: 0.88 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7993855, upper bound: 808.7993546
time: 0.67 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -711.9614258, 577.1731567, -410.5120544, 346.1290894, -1058.0905762, 983.9515381
1: -572.5200806, 560.4498291, -328.9624634, 335.2947083, -907.8148193, 885.9053955
2: -828.7568359, 610.2363892, -478.2442322, 365.9327087, -1194.6892090, 1084.7894287
3: -314.5949707, 801.6986694, -186.3758240, 468.3313599, -779.3368530, 988.0744629
4: -921.2620850, 603.2910767, -533.3006592, 362.9630737, -1284.2249756, 1133.1342773

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6960687, upper bound: 808.1683835
time: 0.68 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6964655, upper bound: 808.6964680
time: 0.81 seconds

## BFS IS instance: IS_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -390.8110657, 326.6912537, -617.0884399, 499.6377258, -886.3889771, 943.7796631
1: -312.7086792, 316.9263611, -495.9963379, 485.3742371, -794.2523193, 812.9227295
2: -453.7768555, 346.5014648, -717.3134766, 528.9486694, -978.8985596, 1063.8148193
3: -176.8959961, 444.4383240, -273.1564026, 692.3924561, -869.2883911, 713.7980347
4: -505.7711182, 343.0486145, -797.0141602, 522.7859497, -1024.9974365, 1140.0627441

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9843982, upper bound: 808.9844013
time: 0.77 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9843982, upper bound: 808.9844013
time: 0.72 seconds

## BFS IS instance: IS_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -426.3378296, 355.6369019, -702.0723877, 568.3237305, -991.3878784, 1057.7092285
1: -341.4696655, 344.8707886, -564.5421143, 551.9707642, -890.2492676, 909.4129028
2: -495.8911438, 376.7834473, -816.9606323, 601.0132446, -1093.6853027, 1193.7440186
3: -192.7451782, 485.1186218, -309.9151001, 790.0589600, -982.8041382, 791.7977295
4: -552.6765137, 373.1866760, -908.1278687, 594.0496216, -1143.8764648, 1281.3145752

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7544494, upper bound: 808.2840274
time: 0.79 seconds

## Relational analysis of IS_B2_A2_B2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9896964, upper bound: 808.9896996
time: 1.02 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.87 seconds
IS_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 4.87
Output dim: 4, lower bound: -803.6075966, upper bound: 803.6075944
IS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 4, lower bound: -803.6075966, upper bound: 807.7369912
IS_B2_A2_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 4.87
Output dim: 4, lower bound: -806.2218297, upper bound: 806.8983947
IS_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 4, lower bound: -808.7993855, upper bound: 808.7993546
IS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 4, lower bound: -808.6960687, upper bound: 808.1683835
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 4, lower bound: -808.6964655, upper bound: 808.6964680
IS_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 4, lower bound: -808.9843982, upper bound: 808.9844013
IS_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 4, lower bound: -808.9843982, upper bound: 808.9844013
IS_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 4, lower bound: -808.7544494, upper bound: 808.2840274
IS_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 4, lower bound: -808.9896964, upper bound: 808.9896996

## BFS IS instance: IS_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -455.2508545, 382.2695007, -451.0949707, 377.2730713, -832.5238647, 833.3644409
1: -364.8853149, 370.5006104, -361.6712952, 365.7339478, -730.6192627, 732.1718750
2: -530.5150757, 404.3332825, -525.7309570, 399.0735779, -929.5885620, 930.0641479
3: -206.0198669, 519.8605957, -203.5385590, 514.5149536, -720.5347900, 723.3991699
4: -591.3572388, 400.4524231, -585.7100830, 395.3730774, -986.7302246, 986.1624756

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 7.10 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.1313355, upper bound: 807.7216201
time: 0.62 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.3885385, upper bound: 807.7222361
time: 0.71 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -388.0713806, 326.9355774, -398.8073425, 335.8592529, -723.9306641, 725.7429199
1: -310.5838623, 317.1121521, -319.4142456, 325.5436096, -636.1274414, 636.5263672
2: -451.1234436, 346.6005859, -464.2101746, 355.4653931, -806.5886841, 810.8107300
3: -177.1954041, 442.0494690, -181.5488586, 454.7042542, -631.8996582, 623.5983276
4: -503.3333130, 343.2300415, -517.7678223, 352.2332764, -855.5665283, 860.9978638

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0848133, upper bound: 808.5304619
time: 0.66 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0847096, upper bound: 808.0847152
time: 0.83 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -691.8927002, 561.2586060, -380.6448669, 324.3557434, -1016.2484131, 937.2947998
1: -556.3126221, 544.9323120, -304.9107361, 313.9728699, -870.2855225, 845.6080933
2: -805.1079102, 593.3786621, -443.3511353, 342.6003113, -1147.7081299, 1031.9554443
3: -306.0691223, 778.9357910, -174.0420837, 435.7907410, -737.5560303, 952.9777832
4: -895.1845093, 586.5603638, -494.7524719, 340.2586060, -1235.4431152, 1076.6813965

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 41

Time for candidate selection: 7.31 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3617485, upper bound: 807.9884286
time: 0.79 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7059003, upper bound: 807.9886834
time: 0.66 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -707.9573364, 574.0289307, -486.4787292, 404.8351440, -1112.7924805, 1056.9229736
1: -569.2694092, 557.3463745, -390.3658447, 392.0812683, -961.3507080, 944.1683960
2: -824.0720215, 606.8655396, -567.1747437, 427.5486755, -1251.6206055, 1170.2186279
3: -312.9233093, 797.1118774, -219.6690674, 553.9096680, -863.1232300, 1016.7809448
4: -916.1106567, 600.0757446, -631.9890747, 424.4866638, -1340.5971680, 1228.5627441

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_B2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5119656, upper bound: 808.0496609
time: 0.75 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9829800, upper bound: 808.6970346
time: 0.67 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -362.4161072, 303.2261353, -617.0607300, 499.6142883, -857.2659912, 920.2868652
1: -290.0926208, 294.3278503, -495.9738159, 485.3514404, -771.0856323, 790.3016357
2: -420.6297302, 321.8147888, -717.2806396, 528.9239502, -944.8870850, 1039.0954590
3: -164.1205139, 412.3122864, -273.1439819, 692.3612061, -856.4816895, 681.1774292
4: -468.4727173, 318.4522095, -796.9777222, 522.7619629, -986.7321777, 1115.4296875

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9839247, upper bound: 808.5120014
time: 0.76 seconds

## Relational analysis of IS_B2_A2_B2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9842404, upper bound: 808.9842434
time: 0.65 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -423.2124634, 352.8832092, -617.0325928, 499.5906372, -917.9288330, 969.9157715
1: -338.9298706, 342.2034912, -495.9511719, 485.3284607, -819.7626953, 838.1545410
2: -492.1878052, 373.9119568, -717.2478027, 528.8991089, -1016.1328125, 1091.1593018
3: -191.2864990, 481.5443115, -273.1314392, 692.3299561, -883.6164551, 750.2374268
4: -548.5905151, 370.3389282, -796.9409790, 522.7378540, -1066.5965576, 1167.2799072

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9839247, upper bound: 808.5120037
time: 0.81 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9842404, upper bound: 808.9842457
time: 0.69 seconds

## BFS IS instance: IS_B2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -361.4206238, 307.8738708, -568.0488281, 467.5066833, -821.9403687, 875.9226074
1: -288.9252014, 298.6079102, -457.0269470, 454.1572571, -737.1547241, 755.6348877
2: -419.6968689, 326.8331604, -661.0649414, 495.1683350, -907.8469238, 987.8980713
3: -166.6037140, 413.0345459, -255.4442596, 642.3858643, -808.9895020, 662.9927979
4: -468.1780090, 323.6291199, -734.6390381, 489.2325745, -950.2793579, 1058.2680664

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_B2_B1_A1

### Relational analysis result of IS_B2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7542969, upper bound: 808.2839479
time: 0.72 seconds

## Relational analysis of IS_B2_A2_B2_B2_B1_A2

### Relational analysis result of IS_B2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3254900, upper bound: 808.2836817
time: 0.78 seconds

## BFS IS instance: IS_B2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -420.5200195, 350.8688049, -691.7791748, 558.4970093, -977.3517456, 1042.6479492
1: -336.8188477, 340.3296509, -556.2202148, 542.6369629, -877.3755493, 896.5498657
2: -489.0038147, 371.8976746, -804.7272339, 590.8292236, -1078.1685791, 1176.6248779
3: -190.2039337, 478.7017212, -304.8158569, 777.6975098, -967.9014282, 780.9042969
4: -545.0430298, 368.2851868, -894.6661987, 583.7861328, -1127.7493896, 1262.9514160

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B2_B2_B2_A1

### Relational analysis result of IS_B2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9600904, upper bound: 806.7604827
time: 0.82 seconds

## Relational analysis of IS_B2_A2_B2_B2_B2_A2

### Relational analysis result of IS_B2_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7604809, upper bound: 806.7604808
time: 0.63 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.76 seconds
IS_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -805.1313355, upper bound: 807.7216201
IS_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -805.3885385, upper bound: 807.7222361
IS_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.0848133, upper bound: 808.5304619
IS_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.0847096, upper bound: 808.0847152
IS_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.3617485, upper bound: 807.9884286
IS_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.7059003, upper bound: 807.9886834
IS_B2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.5119656, upper bound: 808.0496609
IS_B2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.9829800, upper bound: 808.6970346
IS_B2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.9839247, upper bound: 808.5120014
IS_B2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.9842404, upper bound: 808.9842434
IS_B2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.9839247, upper bound: 808.5120037
IS_B2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.9842404, upper bound: 808.9842457
IS_B2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.7542969, upper bound: 808.2839479
IS_B2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.3254900, upper bound: 808.2836817
IS_B2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 4, lower bound: -808.9600904, upper bound: 806.7604827
IS_B2_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 4, lower bound: -806.7604809, upper bound: 806.7604808

## BFS IS instance: IS_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -422.2909851, 358.1248169, -451.0949707, 377.2730713, -799.5640259, 809.2197876
1: -338.0956726, 346.8242188, -361.6712952, 365.7339478, -703.8295288, 708.4954834
2: -491.3074036, 378.7330627, -525.7309570, 399.0735779, -890.3809204, 904.4639893
3: -192.2721863, 482.6188660, -203.5385590, 514.5149536, -706.7871094, 686.1574097
4: -548.5647583, 375.4583740, -585.7100830, 395.3730774, -943.9377441, 961.1684570

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.1312038, upper bound: 807.5132571
time: 0.65 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.1312038, upper bound: 807.7216141
time: 0.81 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -500.6951904, 436.9650879, -451.0949707, 377.2730713, -877.9680786, 888.0599365
1: -401.7972412, 423.6028442, -361.6712952, 365.7339478, -767.5310059, 785.2741089
2: -584.4338379, 462.0734863, -525.7309570, 399.0735779, -983.5073242, 987.8044434
3: -233.9871063, 576.0411987, -203.5385590, 514.5149536, -748.5020752, 779.5797729
4: -651.1557007, 455.8754272, -585.7100830, 395.3730774, -1046.5285645, 1041.5853271

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.3883156, upper bound: 807.5135816
time: 0.77 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.3883156, upper bound: 807.7222301
time: 1.00 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -383.2122803, 322.7327881, -382.9301453, 323.9020081, -707.1141968, 705.6629639
1: -306.6367493, 313.1031189, -306.3933105, 314.0303955, -620.6671143, 619.4963989
2: -445.3221436, 342.2544250, -445.1726685, 342.9752197, -788.2973022, 787.4271240
3: -174.9878693, 436.4235535, -174.9304352, 436.5736389, -611.5614014, 611.3536987
4: -496.9205627, 338.8835754, -496.9640808, 340.0180969, -836.9386597, 835.8475952

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0847096, upper bound: 808.0847152
time: 0.69 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0847096, upper bound: 808.0847152
time: 0.78 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -382.6503296, 322.5522156, -389.3220825, 328.0814819, -710.7318115, 711.8742676
1: -306.1624451, 312.8659363, -311.6675110, 318.0520325, -624.2144165, 624.5333252
2: -444.5317993, 341.9684448, -452.6531982, 347.3631287, -791.8948975, 794.6215210
3: -174.8481293, 435.7823486, -177.4568787, 443.7246704, -618.5728149, 613.2391968
4: -496.0458679, 338.6827087, -504.9850159, 344.2595215, -840.3052979, 843.6677246

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0847096, upper bound: 808.0847152
time: 0.73 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0847096, upper bound: 808.0847152
time: 0.69 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -666.1771240, 541.0292969, -380.6448669, 324.3557434, -990.5328369, 916.7550659
1: -535.5091553, 525.2239380, -304.9107361, 313.9728699, -849.4820557, 825.6597900
2: -774.7716675, 572.1526489, -443.3511353, 342.6003113, -1117.3718262, 1010.4192505
3: -295.2958374, 749.1651611, -174.0420837, 435.7907410, -726.4920044, 923.2071533
4: -862.0064697, 565.6061401, -494.7524719, 340.2586060, -1202.2650146, 1055.3638916

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3616655, upper bound: 807.7443873
time: 0.59 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3616655, upper bound: 807.9884119
time: 0.65 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -760.8027954, 641.5573120, -380.6448669, 324.3557434, -1085.1585693, 1018.2882690
1: -612.2798462, 622.5794067, -304.9107361, 313.9728699, -926.2526855, 923.8209229
2: -886.3291626, 678.0977783, -443.3511353, 342.6003113, -1228.9293213, 1117.3328857
3: -348.0406189, 864.1199951, -174.0420837, 435.7907410, -779.4053955, 1038.1621094
4: -985.8417358, 667.3552246, -494.7524719, 340.2586060, -1326.1000977, 1158.1125488

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7053553, upper bound: 807.7446422
time: 0.68 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7053553, upper bound: 807.9886834
time: 0.72 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -657.7297974, 533.5591431, -411.5172729, 343.7380371, -1001.4678345, 940.1033325
1: -528.6618652, 518.1774292, -329.9764709, 332.9965515, -861.6584473, 843.6094971
2: -765.0318604, 564.4895020, -479.0828552, 363.4306030, -1128.4624023, 1038.5703125
3: -291.1466370, 739.5606689, -185.8440399, 468.8381958, -755.3135376, 925.4047241
4: -850.4497681, 558.1509399, -533.7202148, 360.6166382, -1211.0664062, 1087.0366211

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5119002, upper bound: 808.0495461
time: 0.76 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3314807, upper bound: 808.0494681
time: 0.72 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -704.4957275, 570.9160156, -481.1103821, 399.6217651, -1104.1174316, 1048.3907471
1: -566.4581909, 554.3320312, -386.0127258, 387.1035767, -953.5617676, 936.7385864
2: -819.9710083, 603.5983276, -560.8193359, 422.1637573, -1242.1347656, 1160.4389648
3: -311.2433777, 793.0804443, -217.3868561, 547.5991211, -855.2243652, 1010.4672852
4: -911.5780029, 596.8030396, -624.9537354, 419.1059875, -1330.6839600, 1218.1479492

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9008936, upper bound: 808.6970240
time: 0.76 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9008936, upper bound: 808.6970346
time: 0.99 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -302.9079895, 259.6184387, -490.7644653, 405.3959656, -699.8398438, 750.3829346
1: -241.9817810, 252.1275177, -394.6423950, 393.9593811, -628.8569336, 646.7698975
2: -351.2371826, 276.2890015, -570.4398193, 430.4364014, -773.1689453, 846.7288208
3: -140.1638184, 346.4468384, -222.1657410, 553.8005981, -693.9643555, 562.1317749
4: -391.5410156, 273.2758179, -633.7378540, 424.8596802, -807.5556030, 907.0134888

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B1

### Relational analysis result of IS_B2_A2_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3284171, upper bound: 808.4575506
time: 0.74 seconds

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B2

### Relational analysis result of IS_B2_A2_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3285760, upper bound: 808.3602167
time: 0.62 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -355.4851379, 297.7987366, -606.5888062, 489.5187378, -841.8121948, 904.3875732
1: -284.5224609, 289.1813965, -487.5163879, 475.7188110, -756.9682617, 776.6976929
2: -412.3845520, 316.2391052, -704.8510742, 518.4985962, -927.7448120, 1021.0901489
3: -161.1771698, 404.6792603, -267.5807495, 679.6961060, -840.8732910, 668.8723755
4: -459.3149109, 312.8214722, -783.2244263, 512.2532959, -968.8502808, 1096.0458984

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7902586, upper bound: 808.3286183
time: 0.83 seconds

## Relational analysis of IS_B2_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3284144, upper bound: 808.3284171
time: 0.73 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -357.5846558, 304.6399231, -490.7341919, 405.3704224, -754.2849731, 795.3738403
1: -285.8115540, 295.4838562, -394.6180115, 393.9342957, -672.4337769, 690.1018677
2: -415.1733704, 323.4675903, -570.4041138, 430.4093628, -836.6109009, 893.8717041
3: -164.8842926, 408.6359253, -222.1521301, 553.7664795, -718.6507568, 623.9506836
4: -463.1555176, 320.2887878, -633.6983032, 424.8332520, -878.8489380, 953.9869995

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_B1_A2_B1_B1

### Relational analysis result of IS_B2_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3255018, upper bound: 808.4575671
time: 0.67 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2_B1_B2

### Relational analysis result of IS_B2_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3256606, upper bound: 808.3602333
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -417.4245300, 348.1765747, -606.5590820, 489.4936523, -903.6256104, 954.7354736
1: -334.3058167, 337.7220154, -487.4923096, 475.6944580, -806.5958252, 825.2142334
2: -485.3435059, 369.0863037, -704.8161011, 518.4722900, -1000.3895874, 1073.9023438
3: -188.7670593, 475.1876526, -267.5674744, 679.6626587, -868.4296875, 739.2619629
4: -541.0057983, 365.4977722, -783.1856079, 512.2276001, -1050.2880859, 1148.6833496

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7911525, upper bound: 808.3287200
time: 0.63 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3254991, upper bound: 808.3284337
time: 0.84 seconds

## BFS IS instance: IS_B2_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -343.9009705, 295.1639099, -562.8120117, 462.8173523, -798.7834473, 857.9759521
1: -274.6238403, 286.3842773, -452.7929077, 449.6723328, -717.6342163, 739.1771851
2: -398.9594421, 313.6554260, -654.7949829, 490.2939758, -881.1959229, 968.4503174
3: -159.6189423, 393.5897522, -252.9881287, 636.2230835, -795.8419800, 640.6946411
4: -445.3559265, 310.7164917, -727.7000732, 484.3566284, -921.4251099, 1038.4162598

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7538192, upper bound: 808.2838662
time: 0.72 seconds

## Relational analysis of IS_B2_A2_B2_B2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7538192, upper bound: 808.2839479
time: 0.73 seconds

## BFS IS instance: IS_B2_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -352.1900330, 300.6559448, -563.6359863, 463.8820496, -808.4890747, 864.2919312
1: -281.3902588, 291.6476440, -453.4530945, 450.6613770, -725.6572876, 745.1005249
2: -408.4601135, 319.2653809, -655.7233276, 491.3798828, -892.1028442, 974.9885864
3: -162.7178650, 402.4994507, -253.5277100, 637.2426147, -799.9604492, 650.3114624
4: -455.7279968, 316.1559448, -728.7282715, 485.4804382, -933.2971802, 1044.8841553

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2836534, upper bound: 808.2836533
time: 0.64 seconds

## Relational analysis of IS_B2_A2_B2_B2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2836534, upper bound: 808.2836817
time: 0.82 seconds

## BFS IS instance: IS_B2_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -418.8630676, 349.5828247, -691.7737427, 558.4925537, -975.6625977, 1041.3565674
1: -335.4741516, 339.0836792, -556.2160034, 542.6326294, -876.0034180, 895.2994385
2: -487.0296021, 370.5426331, -804.7210693, 590.8245850, -1076.1649170, 1175.2636719
3: -189.5318604, 476.8672791, -304.8135376, 777.6915283, -967.2233887, 779.0747070
4: -542.8703003, 366.9563293, -894.6593018, 583.7815552, -1125.5477295, 1261.6156006

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7604809, upper bound: 806.7604808
time: 0.74 seconds

## Relational analysis of IS_B2_A2_B2_B2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7604809, upper bound: 806.7604808
time: 0.65 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 9.13 seconds
IS_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -805.1312038, upper bound: 807.5132571
IS_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -805.1312038, upper bound: 807.7216141
IS_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -805.3883156, upper bound: 807.5135816
IS_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -805.3883156, upper bound: 807.7222301
IS_B2_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.0847096, upper bound: 808.0847152
IS_B2_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.0847096, upper bound: 808.0847152
IS_B2_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.0847096, upper bound: 808.0847152
IS_B2_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.0847096, upper bound: 808.0847152
IS_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.3616655, upper bound: 807.7443873
IS_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.3616655, upper bound: 807.9884119
IS_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.7053553, upper bound: 807.7446422
IS_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.7053553, upper bound: 807.9886834
IS_B2_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.5119002, upper bound: 808.0495461
IS_B2_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.3314807, upper bound: 808.0494681
IS_B2_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.9008936, upper bound: 808.6970240
IS_B2_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.9008936, upper bound: 808.6970346
IS_B2_A2_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.3284171, upper bound: 808.4575506
IS_B2_A2_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.3285760, upper bound: 808.3602167
IS_B2_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.7902586, upper bound: 808.3286183
IS_B2_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.3284144, upper bound: 808.3284171
IS_B2_A2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.3255018, upper bound: 808.4575671
IS_B2_A2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.3256606, upper bound: 808.3602333
IS_B2_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.7911525, upper bound: 808.3287200
IS_B2_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.3254991, upper bound: 808.3284337
IS_B2_A2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.7538192, upper bound: 808.2838662
IS_B2_A2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.7538192, upper bound: 808.2839479
IS_B2_A2_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.2836534, upper bound: 808.2836533
IS_B2_A2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 9.13
Output dim: 4, lower bound: -808.2836534, upper bound: 808.2836817
IS_B2_A2_B2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 9.13
Output dim: 4, lower bound: -806.7604809, upper bound: 806.7604808
IS_B2_A2_B2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 9.13
Output dim: 4, lower bound: -806.7604809, upper bound: 806.7604808

## BFS IS instance: IS_B2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -422.2909851, 358.1248169, -418.2478638, 353.3111877, -775.6021729, 776.3726807
1: -338.0956726, 346.8242188, -334.9973145, 342.2789001, -680.3745728, 681.8214111
2: -491.3074036, 378.7330627, -486.7145691, 373.7023621, -865.0096436, 865.4476318
3: -192.2721863, 482.6188660, -189.8989105, 477.4691467, -669.7413330, 672.5176392
4: -548.5647583, 375.4583740, -543.0878906, 370.5376587, -919.1024170, 918.5462646

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.1312038, upper bound: 807.5132571
time: 0.90 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.1312038, upper bound: 807.5132541
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -422.2909851, 358.1248169, -495.3937988, 430.8980713, -853.1889648, 853.5186157
1: -338.0956726, 346.8242188, -397.6154785, 417.7018738, -755.7974243, 744.4395752
2: -491.3074036, 378.7330627, -578.2655640, 455.5121765, -946.8195801, 956.9986572
3: -192.2721863, 482.6188660, -230.8907623, 569.2532349, -761.5253296, 713.5095215
4: -548.5647583, 375.4583740, -643.9641724, 449.7480774, -998.3128052, 1019.4225464

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.1314156, upper bound: 807.7216202
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.1314156, upper bound: 807.7216141
time: 0.93 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -500.6951904, 436.9650879, -418.2478638, 353.3111877, -854.0062256, 855.2129517
1: -401.7972412, 423.6028442, -334.9973145, 342.2789001, -744.0761108, 758.5999146
2: -584.4338379, 462.0734863, -486.7145691, 373.7023621, -958.1361084, 948.7880859
3: -233.9871063, 576.0411987, -189.8989105, 477.4691467, -711.4562378, 765.9401245
4: -651.1557007, 455.8754272, -543.0878906, 370.5376587, -1021.6933594, 998.9633179

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.1312038, upper bound: 807.5132633
time: 0.65 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.3883155, upper bound: 807.5135878
time: 0.76 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -500.6951904, 436.9650879, -495.3937988, 430.8980713, -931.5930176, 932.3588867
1: -401.7972412, 423.6028442, -397.6154785, 417.7018738, -819.4989014, 821.2181396
2: -584.4338379, 462.0734863, -578.2655640, 455.5121765, -1039.9459229, 1040.3391113
3: -233.9871063, 576.0411987, -230.8907623, 569.2532349, -803.2403564, 806.9319458
4: -651.1557007, 455.8754272, -643.9641724, 449.7480774, -1100.9034424, 1099.8394775

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.1314156, upper bound: 807.7216142
time: 0.73 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -805.3886175, upper bound: 807.7222261
time: 0.75 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -372.3106384, 315.1190796, -382.9301453, 323.9020081, -696.2126465, 698.0491943
1: -297.6676025, 305.7345581, -306.3933105, 314.0303955, -611.6979980, 612.1278687
2: -432.2149658, 334.3017883, -445.1726685, 342.9752197, -775.1901855, 779.4744263
3: -170.6806488, 423.9737854, -174.9304352, 436.5736389, -607.2541504, 598.9039307
4: -482.6573486, 331.1230774, -496.9640808, 340.0180969, -822.6754150, 828.0870972

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.8389834, upper bound: 807.8091535
time: 0.75 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0648229, upper bound: 808.4766332
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -378.6096497, 319.2382812, -382.9301453, 323.9020081, -702.5115356, 702.1684570
1: -302.8602295, 309.6569214, -306.3933105, 314.0303955, -616.8906250, 616.0502319
2: -439.5800781, 338.4651489, -445.1726685, 342.9752197, -782.5552979, 783.6378174
3: -173.0794067, 431.0621948, -174.9304352, 436.5736389, -609.6528931, 605.9924316
4: -490.5749817, 335.2482300, -496.9640808, 340.0180969, -830.5930786, 832.2122803

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.8389834, upper bound: 807.8091535
time: 0.74 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0648229, upper bound: 808.4766332
time: 1.01 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -372.3106384, 315.1190796, -389.3220825, 328.0814819, -700.3920898, 704.4411011
1: -297.6676025, 305.7345581, -311.6675110, 318.0520325, -615.7195435, 617.4020386
2: -432.2149658, 334.3017883, -452.6531982, 347.3631287, -779.5780640, 786.9549561
3: -170.6806488, 423.9737854, -177.4568787, 443.7246704, -614.4053345, 601.4303589
4: -482.6573486, 331.1230774, -504.9850159, 344.2595215, -826.9168091, 836.1080933

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.8389325, upper bound: 807.5476013
time: 0.78 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0647514, upper bound: 808.0647571
time: 0.62 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -378.6096497, 319.2382812, -389.3220825, 328.0814819, -706.6911621, 708.5603638
1: -302.8602295, 309.6569214, -311.6675110, 318.0520325, -620.9121094, 621.3244629
2: -439.5800781, 338.4651489, -452.6531982, 347.3631287, -786.9431763, 791.1183472
3: -173.0794067, 431.0621948, -177.4568787, 443.7246704, -616.8040771, 608.5188599
4: -490.5749817, 335.2482300, -504.9850159, 344.2595215, -834.8344727, 840.2332764

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -806.8389325, upper bound: 807.5476013
time: 0.68 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0647514, upper bound: 808.0647571
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -666.1771240, 541.0292969, -361.3013306, 310.3409119, -976.5180664, 897.2551880
1: -535.5091553, 525.2239380, -289.0566406, 300.3057251, -835.8148804, 809.6405640
2: -774.7716675, 572.1526489, -420.0366211, 327.8230591, -1102.5946045, 986.6997070
3: -295.2958374, 749.1651611, -166.2496490, 413.5623474, -704.1567993, 915.4147949
4: -862.0064697, 565.6061401, -469.3685303, 325.7425232, -1187.7490234, 1029.6037598

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3616655, upper bound: 807.7444358
time: 0.64 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3616655, upper bound: 807.7443873
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -666.1771240, 541.0292969, -430.1829224, 383.4602966, -1049.6373291, 965.1848145
1: -535.5091553, 525.2239380, -345.2054443, 371.3411560, -906.8503418, 864.9360352
2: -774.7716675, 572.1526489, -502.6614380, 404.8862305, -1179.6579590, 1068.1068115
3: -295.2958374, 749.1651611, -204.8040924, 498.7010498, -789.7645874, 953.9692383
4: -862.0064697, 565.6061401, -560.5631104, 400.0668335, -1262.0732422, 1119.3569336

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 9
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3617485, upper bound: 807.9884286
time: 0.71 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3617485, upper bound: 807.9884119
time: 0.72 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -760.8027954, 641.5573120, -361.3013306, 310.3409119, -1071.1436768, 998.7885742
1: -612.2798462, 622.5794067, -289.0566406, 300.3057251, -912.5855713, 907.8017578
2: -886.3291626, 678.0977783, -420.0366211, 327.8230591, -1214.1522217, 1093.6134033
3: -348.0406189, 864.1199951, -166.2496490, 413.5623474, -757.0701294, 1030.3696289
4: -985.8417358, 667.3552246, -469.3685303, 325.7425232, -1311.5842285, 1132.3522949

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.6862094, upper bound: 805.9484195
time: 0.66 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7049364, upper bound: 807.6290366
time: 0.61 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -760.8027954, 641.5573120, -430.1829224, 383.4602966, -1144.2629395, 1066.7182617
1: -612.2798462, 622.5794067, -345.2054443, 371.3411560, -983.6209717, 963.0971680
2: -886.3291626, 678.0977783, -502.6614380, 404.8862305, -1291.2153320, 1175.0202637
3: -348.0406189, 864.1199951, -204.8040924, 498.7010498, -842.6778564, 1068.9240723
4: -985.8417358, 667.3552246, -560.5631104, 400.0668335, -1385.9082031, 1222.1057129

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3617485, upper bound: 807.9884119
time: 0.88 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.7059003, upper bound: 807.9886834
time: 0.72 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -639.3316040, 519.6502075, -407.3070984, 339.7828674, -979.1145020, 922.0227661
1: -513.7470093, 504.8620911, -326.5692444, 329.3596191, -843.1066284, 826.9184570
2: -743.2162476, 550.0320435, -474.0584412, 359.5346069, -1102.7507324, 1019.2196045
3: -283.6585693, 719.1228027, -183.8614349, 463.9960938, -742.9323730, 902.9842529
4: -826.4960938, 543.8694458, -528.1342163, 356.7189636, -1183.2148438, 1067.3183594

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5119002, upper bound: 808.0495461
time: 0.81 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5119002, upper bound: 808.0495461
time: 0.69 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -648.3640137, 526.0454102, -406.3560181, 339.5395508, -987.9035645, 927.2354126
1: -521.0535889, 510.9243164, -325.7643738, 328.9454651, -849.9990234, 831.9776001
2: -753.7066040, 556.6152954, -472.8028564, 359.0547180, -1112.7613525, 1024.1384277
3: -287.1715698, 728.7212524, -183.6245422, 462.8889771, -745.3719482, 912.3458252
4: -837.9393311, 550.3917236, -526.7837524, 356.3171387, -1194.2564697, 1072.0657959

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3314807, upper bound: 808.0494681
time: 0.77 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3314807, upper bound: 808.0494681
time: 0.64 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -615.0635376, 498.4917603, -481.1103821, 399.6217651, -1014.6853027, 974.7662964
1: -494.2942200, 484.1149902, -386.0127258, 387.1035767, -881.3978271, 865.5410156
2: -715.0277100, 527.6275635, -560.8193359, 422.1637573, -1137.1911621, 1083.1958008
3: -272.4375305, 690.3264771, -217.3868561, 547.5991211, -815.4899292, 907.7133179
4: -794.6356812, 521.6439819, -624.9537354, 419.1059875, -1213.7415771, 1141.6400146

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5080830, upper bound: 808.1642000
time: 0.80 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5119002, upper bound: 808.2322637
time: 0.69 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3314798, upper bound: 808.2321009
time: 0.82 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -701.0380859, 567.8428955, -481.1103821, 399.6217651, -1100.6599121, 1045.5192871
1: -563.6504517, 551.3801270, -386.0127258, 387.1035767, -950.7540283, 933.9481812
2: -815.8762817, 600.4016724, -560.8193359, 422.1637573, -1238.0400391, 1157.4468994
3: -309.5838623, 789.0825806, -217.3868561, 547.5991211, -853.7274170, 1006.4694214
4: -907.0562744, 593.5829468, -624.9537354, 419.1059875, -1326.1622314, 1215.1497803

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5119002, upper bound: 808.2321802
time: 0.69 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_B2_A2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3314798, upper bound: 808.2319613
time: 0.59 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -296.4460449, 254.1770172, -467.2760925, 387.2607727, -675.1258545, 721.4529419
1: -236.7684784, 246.9208527, -375.5616150, 376.5594482, -606.1692505, 622.4824829
2: -343.5380554, 270.6662903, -542.4406738, 411.6222229, -746.7042236, 813.1068726
3: -137.3227997, 339.0986328, -212.4680634, 527.2400513, -664.5628052, 544.9584351
4: -382.9902039, 267.6575623, -603.0095825, 406.1232300, -780.3090210, 870.6669312

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3284171, upper bound: 808.3602167
time: 0.65 seconds

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3284171, upper bound: 808.3602167
time: 0.74 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -299.5569458, 256.8490601, -483.7799072, 399.6024170, -690.2796631, 740.6289673
1: -239.2517548, 249.4686890, -388.9811096, 388.3741760, -620.2127686, 638.4497070
2: -347.1802673, 273.3870544, -562.0204468, 424.3565369, -762.4832153, 835.4074707
3: -138.6675415, 342.6008606, -219.0927887, 545.5993652, -684.2667847, 555.0087280
4: -387.0711670, 270.4172668, -624.4141235, 418.8655701, -796.5048218, 894.8314209

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B2_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3285760, upper bound: 808.3602167
time: 0.77 seconds

## Relational analysis of IS_B2_A2_B2_B1_A1_B1_B2_A2

### Relational analysis result of IS_B2_A2_B2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3285760, upper bound: 808.3602167
time: 0.73 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -333.2025146, 281.3312073, -601.4410400, 484.9399414, -813.8404541, 882.7722168
1: -266.2678528, 273.2998657, -483.3506775, 471.3486633, -733.4816895, 756.6504517
2: -385.9283142, 299.0889587, -698.7100830, 513.7315063, -895.2334595, 997.7990112
3: -152.1184998, 379.4991760, -265.1782532, 673.6988525, -825.8173218, 640.7564087
4: -430.3391418, 295.9194946, -776.4381714, 507.4674683, -933.6417236, 1072.3576660

Time for backsubstitution: 1.62 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=1011.34521484375
rel_dist={4: [-809.0063734281499, 809.00637342815]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1130.01 seconds
