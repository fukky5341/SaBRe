## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 96.5219627187


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443)
1: (-41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358)
2: (-42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923)
3: (-48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496)
4: (-45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441)

## BASE Result
execution time: IAR + LP analysis = 2.11 + 1.92 = 4.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -96.6185813, upper bound: 96.6185813


# Binary Search by BASE starts (time budget: 1195.97 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=108.20734405517578
rel_dist={4: [-96.61858128162753, 96.61858128162754]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=108.20734405517578
rel_dist={4: [-96.61854736505002, 96.61854736505003]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=108.20734405517578
rel_dist={4: [-96.6182045307215, 96.61820453072153]}

## Binary search (step 3) starts
Candidate diff: 0.0208333


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0208333, mid=0.0208333, abs_max=108.20734405517578
rel_dist={4: [-96.61751649400283, 96.61751649400281]}

## Binary search (step 4) starts
Candidate diff: 0.0104167


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0104167, mid=0.0104167, abs_max=108.20734405517578
rel_dist={4: [-96.61685678066468, 96.61685678066468]}

## Binary search (step 5) starts
Candidate diff: 0.0052083


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0052083, mid=0.0052083, abs_max=108.20734405517578
rel_dist={4: [-96.61647243039103, 96.61647243039101]}

## Binary search (step 6) starts
Candidate diff: 0.0026042


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0026042, mid=0.0026042, abs_max=108.20734405517578
rel_dist={4: [-96.61627320026253, 96.61627320026253]}

## Binary search (step 7) starts
Candidate diff: 0.0013021


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0013021, mid=0.0013021, abs_max=108.20734405517578
rel_dist={4: [-96.61616021813629, 96.61616021813632]}

## Binary search (step 8) starts
Candidate diff: 0.0006510


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0006510, mid=0.0006510, abs_max=108.20734405517578
rel_dist={4: [-96.61610255395536, 96.61610255395533]}

## Binary search (step 9) starts
Candidate diff: 0.0003255


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0003255, mid=0.0003255, abs_max=108.20734405517578
rel_dist={4: [-96.61607302061368, 96.6160730206137]}

## Binary search (step 10) starts
Candidate diff: 0.0001628


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0001628, mid=0.0001628, abs_max=108.20734405517578
rel_dist={4: [-96.61605825396146, 96.61605825396146]}

## Binary search (step 11) starts
Candidate diff: 0.0000814


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000814, mid=0.0000814, abs_max=108.20734405517578
rel_dist={4: [-96.6160508706725, 96.6160508706725]}

## Binary search (step 12) starts
Candidate diff: 0.0000407


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000407, mid=0.0000407, abs_max=108.20734405517578
rel_dist={4: [-96.61604717910173, 96.61604717910171]}

## Binary search (step 13) starts
Candidate diff: 0.0000203


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000203, mid=0.0000203, abs_max=108.20734405517578
rel_dist={4: [-96.61604533346147, 96.6160453334615]}

## Binary search (step 14) starts
Candidate diff: 0.0000102


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000102, mid=0.0000102, abs_max=108.20734405517578
rel_dist={4: [-96.61604441092287, 96.61604441092288]}

## Binary search (step 15) starts
Candidate diff: 0.0000051


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000051, mid=0.0000051, abs_max=108.20734405517578
rel_dist={4: [-96.61604395018364, 96.61604395018367]}

## Binary search (step 16) starts
Candidate diff: 0.0000025


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000025, mid=0.0000025, abs_max=108.20734405517578
rel_dist={4: [-96.61604505545549, 96.61604424113546]}

## Binary search (step 17) starts
Candidate diff: 0.0000013


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000013, mid=0.0000013, abs_max=108.20734405517578
rel_dist={4: [-96.6160441367285, 96.61604384909046]}

## Binary search (step 18) starts
Candidate diff: 0.0000006


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000006, mid=0.0000006, abs_max=108.20734405517578
rel_dist={4: [-96.61604369715752, 96.6160437804491]}

## Binary Search Result
Binary search time: 81.38 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1114.59 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5315669, upper bound: 96.6017281
time: 0.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6152408, upper bound: 96.6152409
time: 0.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.86
Output dim: 4, lower bound: -96.5315669, upper bound: 96.6017281
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.86
Output dim: 4, lower bound: -96.6152408, upper bound: 96.6152409

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -23.6491947, 44.5811996, -38.2100830, 62.7565804, -86.4057770, 82.7912750
1: -25.9020424, 36.9293671, -41.6995087, 54.3409386, -80.2429733, 78.6288757
2: -26.6062851, 36.6187668, -42.7050247, 54.2436867, -80.8499603, 79.3237762
3: -30.8861237, 42.7392807, -48.9462700, 63.1773834, -94.0635071, 91.6855469
4: -29.5871983, 41.9198265, -45.2550850, 62.9522591, -92.5394592, 87.1749115

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5180541, upper bound: 96.5180541
time: 0.89 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5180541, upper bound: 96.6017281
time: 1.07 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -47.7588348, 79.9911652, -38.0210800, 62.4870491, -110.2458725, 118.0122452
1: -52.1639366, 67.9121017, -41.4938278, 54.0912552, -106.2551804, 109.4059219
2: -53.4510193, 68.0761642, -42.4946556, 53.9964714, -107.4474945, 110.5708160
3: -61.4133224, 78.7076035, -48.7111588, 62.8709106, -124.2842178, 127.4187622
4: -56.2418251, 79.0030441, -45.0321922, 62.6598892, -118.9017105, 124.0352325

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6017281, upper bound: 96.5315669
time: 1.04 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6017281, upper bound: 96.6152409
time: 0.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.48 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 4.48
Output dim: 4, lower bound: -96.5180541, upper bound: 96.5180541
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 4, lower bound: -96.5180541, upper bound: 96.6017281
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 4, lower bound: -96.6017281, upper bound: 96.5315669
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 4, lower bound: -96.6017281, upper bound: 96.6152409

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -23.6491947, 44.5811996, -47.7588348, 79.9911652, -103.6403580, 92.3400269
1: -25.9020424, 36.9293671, -52.1639366, 67.9121017, -93.8141251, 89.0933075
2: -26.6062851, 36.6187668, -53.4510193, 68.0761642, -94.6824493, 90.0697861
3: -30.8861237, 42.7392807, -61.4133224, 78.7076035, -109.5937271, 104.1525955
4: -29.5871983, 41.9198265, -56.2418251, 79.0030441, -108.5902405, 98.1616516

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4918641, upper bound: 96.5901099
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5178728, upper bound: 96.5178728
time: 0.80 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -47.7588348, 79.9911652, -23.6491947, 44.5811996, -92.3400269, 103.6403580
1: -52.1639366, 67.9121017, -25.9020424, 36.9293671, -89.0933075, 93.8141251
2: -53.4510193, 68.0761642, -26.6062851, 36.6187668, -90.0697784, 94.6824493
3: -61.4133224, 78.7076035, -30.8861237, 42.7392807, -104.1525955, 109.5937271
4: -56.2418251, 79.0030441, -29.5871983, 41.9198265, -98.1616516, 108.5902405

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5558460, upper bound: 96.5172514
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6015467, upper bound: 96.5315373
time: 0.97 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -47.7588348, 79.9911652, -47.7588348, 79.9911652, -127.7499771, 127.7499771
1: -52.1639366, 67.9121017, -52.1639366, 67.9121017, -120.0760345, 120.0760345
2: -53.4510193, 68.0761642, -53.4510193, 68.0761642, -121.5271835, 121.5271835
3: -61.4133224, 78.7076035, -61.4133224, 78.7076035, -140.1209106, 140.1208954
4: -56.2418251, 79.0030441, -56.2418251, 79.0030441, -135.2448730, 135.2448730

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5558461, upper bound: 96.5981244
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6015468, upper bound: 96.6152112
time: 0.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.24 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 4, lower bound: -96.4918641, upper bound: 96.5901099
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.24
Output dim: 4, lower bound: -96.5178728, upper bound: 96.5178728
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 4, lower bound: -96.5558460, upper bound: 96.5172514
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 4, lower bound: -96.6015467, upper bound: 96.5315373
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 4, lower bound: -96.5558461, upper bound: 96.5981244
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 4, lower bound: -96.6015468, upper bound: 96.6152112

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -19.0168400, 37.8047943, -47.7588348, 79.9911652, -99.0079956, 85.5636215
1: -20.9385338, 31.1538448, -52.1639366, 67.9121017, -88.8506241, 83.3177795
2: -21.5286846, 30.8517284, -53.4510193, 68.0761642, -89.6048508, 84.3027496
3: -25.2066669, 36.0842285, -61.4133224, 78.7076035, -103.9142685, 97.4975204
4: -24.6514683, 35.1617165, -56.2418251, 79.0030441, -103.6545029, 91.4035416

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4912427, upper bound: 96.5444407
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4912427, upper bound: 96.5901098
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -41.6879997, 71.5514069, -23.6491947, 44.5811996, -86.2691956, 95.2005920
1: -45.5929909, 60.5228043, -25.9020424, 36.9293671, -82.5223541, 86.4248428
2: -46.7689133, 60.5786438, -26.6062851, 36.6187668, -83.3876648, 87.1849213
3: -53.8531685, 70.0829163, -30.8861237, 42.7392807, -96.5924454, 100.9690399
4: -49.6292267, 70.1049042, -29.5871983, 41.9198265, -91.5490494, 99.6921005

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5444407, upper bound: 96.4912427
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5444407, upper bound: 96.5172513
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -47.0677299, 79.0901794, -23.6491947, 44.5811996, -91.6489258, 102.7393723
1: -51.4176712, 67.0940704, -25.9020424, 36.9293671, -88.3470383, 92.9961090
2: -52.6958504, 67.2510605, -26.6062851, 36.6187668, -89.3145981, 93.8573303
3: -60.5642242, 77.7514114, -30.8861237, 42.7392807, -103.3035049, 108.6375351
4: -55.4830666, 78.0209122, -29.5871983, 41.9198265, -97.4028931, 107.6081085

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5901098, upper bound: 96.5054049
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5901098, upper bound: 96.5315373
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -41.6879997, 71.5514069, -47.7588348, 79.9911652, -121.6791611, 119.3102341
1: -45.5929909, 60.5228043, -52.1639366, 67.9121017, -113.5050888, 112.6867371
2: -46.7689133, 60.5786438, -53.4510193, 68.0761642, -114.8450775, 114.0296631
3: -53.8531685, 70.0829163, -61.4133224, 78.7076035, -132.5607758, 131.4962311
4: -49.6292267, 70.1049042, -56.2418251, 79.0030441, -128.6322632, 126.3467255

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5552561, upper bound: 96.5551778
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5552561, upper bound: 96.5551778
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -47.0677299, 79.0901794, -47.7588348, 79.9911652, -127.0588837, 126.8489914
1: -51.4176712, 67.0940704, -52.1639366, 67.9121017, -119.3297729, 119.2580109
2: -52.6958504, 67.2510605, -53.4510193, 68.0761642, -120.7720184, 120.7020645
3: -60.5642242, 77.7514114, -61.4133224, 78.7076035, -139.2718201, 139.1647186
4: -55.4830666, 78.0209122, -56.2418251, 79.0030441, -134.4861145, 134.2627411

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6009253, upper bound: 96.5690058
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6009253, upper bound: 96.5690058
time: 0.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.36 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -96.4912427, upper bound: 96.5444407
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -96.4912427, upper bound: 96.5901098
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -96.5444407, upper bound: 96.4912427
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -96.5444407, upper bound: 96.5172513
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -96.5901098, upper bound: 96.5054049
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -96.5901098, upper bound: 96.5315373
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -96.5552561, upper bound: 96.5551778
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -96.5552561, upper bound: 96.5551778
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -96.6009253, upper bound: 96.5690058
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 4, lower bound: -96.6009253, upper bound: 96.5690058

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -19.0168400, 37.8047943, -41.6879997, 71.5514069, -90.5682449, 79.4927979
1: -20.9385338, 31.1538448, -45.5929909, 60.5228043, -81.4613342, 76.7468338
2: -21.5286846, 30.8517284, -46.7689133, 60.5786438, -82.1073303, 77.6206436
3: -25.2066669, 36.0842285, -53.8531685, 70.0829163, -95.2895813, 89.9373703
4: -24.6514683, 35.1617165, -49.6292267, 70.1049042, -94.7563705, 84.7909393

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4284456, upper bound: 96.5328962
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4769927, upper bound: 96.5416326
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -19.0168400, 37.8047943, -47.0677299, 79.0901794, -98.1070175, 84.8725281
1: -20.9385338, 31.1538448, -51.4176712, 67.0940704, -88.0326080, 82.5715179
2: -21.5286846, 30.8517284, -52.6958504, 67.2510605, -88.7797470, 83.5475769
3: -25.2066669, 36.0842285, -60.5642242, 77.7514114, -102.9580765, 96.6484375
4: -24.6514683, 35.1617165, -55.4830666, 78.0209122, -102.6723785, 90.6447830

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4284456, upper bound: 96.5751409
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4769927, upper bound: 96.5873018
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -41.6879997, 71.5514069, -19.0168400, 37.8047943, -79.4927979, 90.5682449
1: -45.5929909, 60.5228043, -20.9385338, 31.1538448, -76.7468338, 81.4613342
2: -46.7689133, 60.5786438, -21.5286846, 30.8517284, -77.6206436, 82.1073303
3: -53.8531685, 70.0829163, -25.2066669, 36.0842285, -89.9373779, 95.2895813
4: -49.6292267, 70.1049042, -24.6514683, 35.1617165, -84.7909393, 94.7563629

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5326938, upper bound: 96.4525539
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5411443, upper bound: 96.4868457
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5284280, upper bound: 96.4463417
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5313152, upper bound: 96.4868711
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5434006, upper bound: 96.4889566
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -41.6879997, 71.5514069, -22.9655075, 43.7550316, -85.4430313, 94.5169144
1: -45.5929909, 60.5228043, -25.1754646, 36.1651459, -81.7581329, 85.6982651
2: -46.7689133, 60.5786438, -25.8641987, 35.8516693, -82.6205750, 86.4428253
3: -53.8531685, 70.0829163, -30.0730324, 41.8423233, -95.6954956, 100.1559448
4: -49.6292267, 70.1049042, -28.8824005, 41.0101280, -90.6393433, 98.9873047

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5326938, upper bound: 96.4956587
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5411443, upper bound: 96.5171582
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5284280, upper bound: 96.4463417
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5313152, upper bound: 96.5068631
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5434006, upper bound: 96.5089486
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -47.0677299, 79.0901794, -19.0168400, 37.8047943, -84.8725281, 98.1070175
1: -51.4176712, 67.0940704, -20.9385338, 31.1538448, -82.5715179, 88.0326080
2: -52.6958504, 67.2510605, -21.5286846, 30.8517284, -83.5475693, 88.7797470
3: -60.5642242, 77.7514114, -25.2066669, 36.0842285, -96.6484451, 102.9580765
4: -55.4830666, 78.0209122, -24.6514683, 35.1617165, -90.6447830, 102.6723709

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4985212, upper bound: 96.4896334
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5898532, upper bound: 96.5048780
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -47.0677299, 79.0901794, -22.9655075, 43.7550316, -90.8227386, 102.0556870
1: -51.4176712, 67.0940704, -25.1754646, 36.1651459, -87.5828171, 92.2695312
2: -52.6958504, 67.2510605, -25.8641987, 35.8516693, -88.5475082, 93.1152344
3: -60.5642242, 77.7514114, -30.0730324, 41.8423233, -102.4065475, 107.8244476
4: -55.4830666, 78.0209122, -28.8824005, 41.0101280, -96.4931946, 106.9033051

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5502077, upper bound: 96.5192467
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5898532, upper bound: 96.5262182
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -41.6879997, 71.5514069, -41.6879997, 71.5514069, -113.2394104, 113.2394028
1: -45.5929909, 60.5228043, -45.5929909, 60.5228043, -106.1157990, 106.1157990
2: -46.7689133, 60.5786438, -46.7689133, 60.5786438, -107.3475494, 107.3475571
3: -53.8531685, 70.0829163, -53.8531685, 70.0829163, -123.9360809, 123.9360809
4: -49.6292267, 70.1049042, -49.6292267, 70.1049042, -119.7341309, 119.7341309

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5413631, upper bound: 96.5482585
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544929, upper bound: 96.5544234
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -41.6879997, 71.5514069, -47.0677299, 79.0901794, -120.7781830, 118.6191254
1: -45.5929909, 60.5228043, -51.4176712, 67.0940704, -112.6870575, 111.9404755
2: -46.7689133, 60.5786438, -52.6958504, 67.2510605, -114.0199585, 113.2744904
3: -53.8531685, 70.0829163, -60.5642242, 77.7514114, -131.6045685, 130.6471405
4: -49.6292267, 70.1049042, -55.4830666, 78.0209122, -127.6501389, 125.5879669

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5413631, upper bound: 96.5743868
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544929, upper bound: 96.5969939
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -47.0677299, 79.0901794, -41.6879997, 71.5514069, -118.6191406, 120.7781754
1: -51.4176712, 67.0940704, -45.5929909, 60.5228043, -111.9404755, 112.6870575
2: -52.6958504, 67.2510605, -46.7689133, 60.5786438, -113.2744827, 114.0199585
3: -60.5642242, 77.7514114, -53.8531685, 70.0829163, -130.6471405, 131.6045837
4: -55.4830666, 78.0209122, -49.6292267, 70.1049042, -125.5879669, 127.6501389

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5496191, upper bound: 96.5502421
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6008946, upper bound: 96.5686156
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -47.0677299, 79.0901794, -47.0677299, 79.0901794, -126.1579056, 126.1578827
1: -51.4176712, 67.0940704, -51.4176712, 67.0940704, -118.5117416, 118.5117416
2: -52.6958504, 67.2510605, -52.6958504, 67.2510605, -119.9468994, 119.9468994
3: -60.5642242, 77.7514114, -60.5642242, 77.7514114, -138.3156433, 138.3156433
4: -55.4830666, 78.0209122, -55.4830666, 78.0209122, -133.5039825, 133.5039825

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5496191, upper bound: 96.5885899
time: 1.33 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6008946, upper bound: 96.6147527
time: 1.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.66 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.4284456, upper bound: 96.5328962
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.4769927, upper bound: 96.5416326
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.4284456, upper bound: 96.5751409
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.4769927, upper bound: 96.5873018
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5313152, upper bound: 96.4868711
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5434006, upper bound: 96.4889566
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5313152, upper bound: 96.5068631
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5434006, upper bound: 96.5089486
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.4985212, upper bound: 96.4896334
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5898532, upper bound: 96.5048780
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5502077, upper bound: 96.5192467
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5898532, upper bound: 96.5262182
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5413631, upper bound: 96.5482585
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5544929, upper bound: 96.5544234
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5413631, upper bound: 96.5743868
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5544929, upper bound: 96.5969939
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5496191, upper bound: 96.5502421
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.6008946, upper bound: 96.5686156
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.5496191, upper bound: 96.5885899
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.66
Output dim: 4, lower bound: -96.6008946, upper bound: 96.6147527

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.9556608, 31.9021549, -41.2109756, 70.9338760, -85.8895264, 73.1131134
1: -16.5861988, 26.0356140, -45.0696716, 59.9224205, -76.5085983, 71.1052856
2: -17.0159492, 25.7218666, -46.2414360, 59.9695282, -76.9854736, 71.9633026
3: -20.1648331, 30.1525993, -53.2424660, 69.3777466, -89.5425797, 83.3950577
4: -20.2020149, 29.1788654, -49.0922356, 69.3751984, -89.5772095, 78.2711029

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4229860, upper bound: 96.5258743
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4284456, upper bound: 96.5304470
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4188454, upper bound: 96.5205231
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.3886322, upper bound: 96.5106698
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.3902581, upper bound: 96.5212828
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -18.3205643, 36.8871384, -41.6879997, 71.5514069, -89.8719635, 78.5751343
1: -20.1822853, 30.2803841, -45.5929909, 60.5228043, -80.7050934, 75.8733597
2: -20.7637596, 29.9831238, -46.7689133, 60.5786438, -81.3423996, 76.7520294
3: -24.3498306, 35.0611115, -53.8531685, 70.0829163, -94.4327469, 88.9142685
4: -23.8476677, 34.1320457, -49.6292267, 70.1049042, -93.9525757, 83.7612762

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4701123, upper bound: 96.5378411
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4718376, upper bound: 96.5275822
time: 4.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4743223, upper bound: 96.5405982
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.9556608, 31.9021549, -46.4869156, 78.3419800, -93.2976379, 78.3890610
1: -16.5861988, 26.0356140, -50.7803459, 66.3616791, -82.9478683, 76.8159561
2: -17.0159492, 25.7218666, -52.0477600, 66.5163574, -83.5323029, 77.7696228
3: -20.1648331, 30.1525993, -59.8246002, 76.8819427, -97.0467758, 89.9772034
4: -20.2020149, 29.1788654, -54.8062782, 77.1366730, -97.3386841, 83.9851303

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4121841, upper bound: 96.5237999
time: 2.09 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4202881, upper bound: 96.5677775
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -18.3205643, 36.8871384, -47.0677299, 79.0901794, -97.4107437, 83.9548492
1: -20.1822853, 30.2803841, -51.4176712, 67.0940704, -87.2763519, 81.6980591
2: -20.7637596, 29.9831238, -52.6958504, 67.2510605, -88.0148163, 82.6789703
3: -24.3498306, 35.0611115, -60.5642242, 77.7514114, -102.1012421, 95.6253281
4: -23.8476677, 34.1320457, -55.4830666, 78.0209122, -101.8685760, 89.6151123

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4817290, upper bound: 96.5432885
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4902744, upper bound: 96.5870016
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -41.3976822, 71.2384109, -19.0168400, 37.8047943, -79.2024689, 90.2552414
1: -45.2824707, 60.2273254, -20.9385338, 31.1538448, -76.4363174, 81.1658325
2: -46.4568481, 60.2715683, -21.5286846, 30.8517284, -77.3085785, 81.8002472
3: -53.5106544, 69.7326126, -25.2066669, 36.0842285, -89.5948792, 94.9392776
4: -49.3290863, 69.7293549, -24.6514683, 35.1617165, -84.4907990, 94.3808212

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5106698, upper bound: 96.3886322
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5275822, upper bound: 96.4718375
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -41.5644608, 71.5200424, -19.0150261, 37.8027878, -79.3672485, 90.5350647
1: -45.4678535, 60.4032860, -20.9366531, 31.1519032, -76.6197510, 81.3399200
2: -46.6354332, 60.4574471, -21.5267124, 30.8497543, -77.4851761, 81.9841537
3: -53.7209549, 69.9424973, -25.2046223, 36.0819588, -89.8029175, 95.1471176
4: -49.5006561, 69.9765320, -24.6496277, 35.1594276, -84.6600800, 94.6261597

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5212828, upper bound: 96.3902581
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5405982, upper bound: 96.4743222
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -41.3976822, 71.2384109, -22.9655075, 43.7550316, -85.1526947, 94.2039108
1: -45.2824707, 60.2273254, -25.1754646, 36.1651459, -81.4476166, 85.4027710
2: -46.4568481, 60.2715683, -25.8641987, 35.8516693, -82.3085098, 86.1357422
3: -53.5106544, 69.7326126, -30.0730324, 41.8423233, -95.3529816, 99.8056488
4: -49.3290863, 69.7293549, -28.8824005, 41.0101280, -90.3392105, 98.6117554

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5377706, upper bound: 96.5049930
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5342520, upper bound: 96.4971075
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -41.5644608, 71.5200424, -22.9636860, 43.7531204, -85.3175812, 94.4837189
1: -45.4678535, 60.4032860, -25.1735649, 36.1632843, -81.6311188, 85.5768280
2: -46.6354332, 60.4574471, -25.8622303, 35.8497849, -82.4852142, 86.3196716
3: -53.7209549, 69.9424973, -30.0709591, 41.8401299, -95.5610809, 100.0134506
4: -49.5006561, 69.9765320, -28.8805313, 41.0079079, -90.5085449, 98.8570633

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5498560, upper bound: 96.5070785
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5463374, upper bound: 96.4991930
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -45.9752731, 78.6173096, -19.0168400, 37.8047943, -83.7800674, 97.6341476
1: -50.2814026, 66.3871689, -20.9385338, 31.1538448, -81.4352493, 87.3256989
2: -51.5448151, 66.4945831, -21.5286846, 30.8517284, -82.3965302, 88.0232697
3: -59.4520378, 76.8625641, -25.2066669, 36.0842285, -95.5362473, 102.0692291
4: -54.4311790, 77.0660400, -24.6514683, 35.1617165, -89.5928955, 101.7174988

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5885210, upper bound: 96.5048780
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5677775, upper bound: 96.4202881
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5870016, upper bound: 96.4902744
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -44.7436600, 76.0611954, -22.9655075, 43.7550316, -88.4986877, 99.0266953
1: -48.8998871, 64.2461166, -25.1754646, 36.1651459, -85.0650177, 89.4215851
2: -50.1368523, 64.3910141, -25.8641987, 35.8516693, -85.9884949, 90.2551956
3: -57.7000580, 74.4001617, -30.0730324, 41.8423233, -99.5423813, 104.4731903
4: -52.8640900, 74.5853195, -28.8824005, 41.0101280, -93.8742065, 103.4677124

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5604932, upper bound: 96.5192467
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5605658, upper bound: 96.5182960
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -45.9752731, 78.6173096, -22.9655075, 43.7550316, -89.7303009, 101.5828171
1: -50.2814026, 66.3871689, -25.1754646, 36.1651459, -86.4465485, 91.5626373
2: -51.5448151, 66.4945831, -25.8641987, 35.8516693, -87.3964691, 92.3587799
3: -59.4520378, 76.8625641, -30.0730324, 41.8423233, -101.2943573, 106.9355850
4: -54.4311790, 77.0660400, -28.8824005, 41.0101280, -95.4413071, 105.9484329

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5976788, upper bound: 96.5262182
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977514, upper bound: 96.5252675
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -41.3976822, 71.2384109, -41.6879997, 71.5514069, -112.9490814, 112.9264069
1: -45.2824707, 60.2273254, -45.5929909, 60.5228043, -105.8052750, 105.8202972
2: -46.4568481, 60.2715683, -46.7689133, 60.5786438, -107.0354767, 107.0404816
3: -53.5106544, 69.7326126, -53.8531685, 70.0829163, -123.5935669, 123.5857849
4: -49.3290863, 69.7293549, -49.6292267, 70.1049042, -119.4339905, 119.3585815

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5403220, upper bound: 96.5403220
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5403220, upper bound: 96.5403220
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -41.5644608, 71.5200424, -41.6860123, 71.5490265, -113.1134872, 113.2060471
1: -45.4678535, 60.4032860, -45.5908470, 60.5205116, -105.9883652, 105.9941177
2: -46.6354332, 60.4574471, -46.7667198, 60.5763359, -107.2117538, 107.2241669
3: -53.7209549, 69.9424973, -53.8507462, 70.0802765, -123.8012238, 123.7932434
4: -49.5006561, 69.9765320, -49.6270218, 70.1021729, -119.6028214, 119.6035538

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5524079, upper bound: 96.5424076
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5524079, upper bound: 96.5544235
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -41.3976822, 71.2384109, -47.0677299, 79.0901794, -120.4878540, 118.3061066
1: -45.2824707, 60.2273254, -51.4176712, 67.0940704, -112.3765411, 111.6449890
2: -46.4568481, 60.2715683, -52.6958504, 67.2510605, -113.7078857, 112.9674225
3: -53.5106544, 69.7326126, -60.5642242, 77.7514114, -131.2620697, 130.2968445
4: -49.3290863, 69.7293549, -55.4830666, 78.0209122, -127.3499985, 125.2124176

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5410159, upper bound: 96.5206366
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5512513, upper bound: 96.5739679
time: 1.42 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -41.5644608, 71.5200424, -47.0655518, 79.0876007, -120.6520538, 118.5855942
1: -45.4678535, 60.4032860, -51.4153214, 67.0916061, -112.5594635, 111.8185959
2: -46.6354332, 60.4574471, -52.6934738, 67.2485352, -113.8839569, 113.1509094
3: -53.7209549, 69.9424973, -60.5615540, 77.7485886, -131.4695282, 130.5040588
4: -49.5006561, 69.9765320, -55.4807587, 78.0178909, -127.5185318, 125.4572754

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5531013, upper bound: 96.5227221
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5661198, upper bound: 96.5969939
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -44.7436600, 76.0611954, -41.6879997, 71.5514069, -116.2950592, 117.7491913
1: -48.8998871, 64.2461166, -45.5929909, 60.5228043, -109.4226913, 109.8391113
2: -50.1368523, 64.3910141, -46.7689133, 60.5786438, -110.7154770, 111.1599197
3: -57.7000580, 74.4001617, -53.8531685, 70.0829163, -127.7829742, 128.2533264
4: -52.8640900, 74.5853195, -49.6292267, 70.1049042, -122.9689865, 124.2145462

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5495550, upper bound: 96.5502281
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5486094, upper bound: 96.5497029
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5066818, upper bound: 96.5063777
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5496188, upper bound: 96.5500946
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5490962, upper bound: 96.5224362
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -45.9752731, 78.6173096, -41.6879997, 71.5514069, -117.5266724, 120.3053055
1: -50.2814026, 66.3871689, -45.5929909, 60.5228043, -110.8042068, 111.9801559
2: -51.5448151, 66.4945831, -46.7689133, 60.5786438, -112.1234436, 113.2634964
3: -59.4520378, 76.8625641, -53.8531685, 70.0829163, -129.5349579, 130.7157288
4: -54.4311790, 77.0660400, -49.6292267, 70.1049042, -124.5360870, 126.6952667

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5994868, upper bound: 96.5684090
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5914318, upper bound: 96.5651618
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5976253, upper bound: 96.5558807
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5997715, upper bound: 96.5678397
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -44.7436600, 76.0611954, -47.0677299, 79.0901794, -123.8338394, 123.1289215
1: -48.8998871, 64.2461166, -51.4176712, 67.0940704, -115.9939575, 115.6637878
2: -50.1368523, 64.3910141, -52.6958504, 67.2510605, -117.3878937, 117.0868530
3: -57.7000580, 74.4001617, -60.5642242, 77.7514114, -135.4514618, 134.9643860
4: -52.8640900, 74.5853195, -55.4830666, 78.0209122, -130.8849945, 130.0683746

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5714769, upper bound: 96.5885899
time: 1.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5724615, upper bound: 96.5885127
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -45.9752731, 78.6173096, -47.0677299, 79.0901794, -125.0654526, 125.6850128
1: -50.2814026, 66.3871689, -51.4176712, 67.0940704, -117.3754654, 117.8048325
2: -51.5448151, 66.4945831, -52.6958504, 67.2510605, -118.7958450, 119.1904297
3: -59.4520378, 76.8625641, -60.5642242, 77.7514114, -137.2034454, 137.4267883
4: -54.4311790, 77.0660400, -55.4830666, 78.0209122, -132.4520874, 132.5491028

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6120489, upper bound: 96.6130762
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6128915, upper bound: 96.6134205
time: 1.16 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.41 seconds
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.3886322, upper bound: 96.5106698
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.3902581, upper bound: 96.5212828
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.4718376, upper bound: 96.5275822
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.4743223, upper bound: 96.5405982
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.4121841, upper bound: 96.5237999
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.4202881, upper bound: 96.5677775
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.4817290, upper bound: 96.5432885
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.4902744, upper bound: 96.5870016
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5106698, upper bound: 96.3886322
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5275822, upper bound: 96.4718375
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5212828, upper bound: 96.3902581
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5405982, upper bound: 96.4743222
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5377706, upper bound: 96.5049930
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5342520, upper bound: 96.4971075
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5498560, upper bound: 96.5070785
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5463374, upper bound: 96.4991930
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5677775, upper bound: 96.4202881
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5870016, upper bound: 96.4902744
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5604932, upper bound: 96.5192467
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5605658, upper bound: 96.5182960
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5976788, upper bound: 96.5262182
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5977514, upper bound: 96.5252675
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5403220, upper bound: 96.5403220
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5403220, upper bound: 96.5403220
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5524079, upper bound: 96.5424076
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5524079, upper bound: 96.5544235
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5410159, upper bound: 96.5206366
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5512513, upper bound: 96.5739679
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5531013, upper bound: 96.5227221
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5661198, upper bound: 96.5969939
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5496188, upper bound: 96.5500946
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5490962, upper bound: 96.5224362
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5976253, upper bound: 96.5558807
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5997715, upper bound: 96.5678397
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5714769, upper bound: 96.5885899
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.5724615, upper bound: 96.5885127
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.6120489, upper bound: 96.6130762
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 4, lower bound: -96.6128915, upper bound: 96.6134205

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -18.3205643, 36.8871384, -41.3976822, 71.2384109, -89.5589676, 78.2847977
1: -20.1822853, 30.2803841, -45.2824707, 60.2273254, -80.4096069, 75.5628510
2: -20.7637596, 29.9831238, -46.4568481, 60.2715683, -81.0353165, 76.4399643
3: -24.3498306, 35.0611115, -53.5106544, 69.7326126, -94.0824432, 88.5717621
4: -23.8476677, 34.1320457, -49.3290863, 69.7293549, -93.5770264, 83.4611359

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4701458, upper bound: 96.5272838
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4718376, upper bound: 96.5275385
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -18.3188400, 36.8851891, -41.5644608, 71.5200424, -89.8388824, 78.4496307
1: -20.1804886, 30.2785072, -45.4678535, 60.4032860, -80.5837631, 75.7463531
2: -20.7618847, 29.9812012, -46.6354332, 60.4574471, -81.2193146, 76.6166306
3: -24.3478432, 35.0588989, -53.7209549, 69.9424973, -94.2903366, 88.7798462
4: -23.8458786, 34.1297989, -49.5006561, 69.9765320, -93.8223953, 83.6304321

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4726304, upper bound: 96.5402998
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4743223, upper bound: 96.5405546
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.9556608, 31.9021549, -44.1661797, 75.3191757, -90.2748337, 76.0683365
1: -16.5861988, 26.0356140, -48.2666702, 63.5292320, -80.1154099, 74.3022766
2: -17.0159492, 25.7218666, -49.4982109, 63.6598091, -80.6757507, 75.2200623
3: -20.1648331, 30.1525993, -56.9633675, 73.5578690, -93.7227020, 87.1159668
4: -20.2020149, 29.1788654, -52.2160721, 73.7083054, -93.9103165, 81.3949051

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4121841, upper bound: 96.5237999
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4116028, upper bound: 96.5210880
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4052172, upper bound: 96.4657995
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.3486577, upper bound: 96.4462652
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4073714, upper bound: 96.5227888
time: 1.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.9556608, 31.9021549, -45.4107056, 77.8863297, -92.8419876, 77.3128510
1: -16.5861988, 26.0356140, -49.6619873, 65.6825562, -82.2687531, 75.6976013
2: -17.0159492, 25.7218666, -50.9203415, 65.7741852, -82.7901306, 76.6422043
3: -20.1648331, 30.1525993, -58.7286339, 76.0346375, -96.1994705, 88.8812332
4: -20.2020149, 29.1788654, -53.7956123, 76.2025909, -96.4046021, 82.9744720

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4202881, upper bound: 96.5677775
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4200252, upper bound: 96.5667056
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4156994, upper bound: 96.5657105
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.3917805, upper bound: 96.5586795
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -18.3205643, 36.8871384, -44.7436600, 76.0611954, -94.3817520, 81.6307831
1: -20.1822853, 30.2803841, -48.8998871, 64.2461166, -84.4284058, 79.1802521
2: -20.7637596, 29.9831238, -50.1368523, 64.3910141, -85.1547699, 80.1199646
3: -24.3498306, 35.0611115, -57.7000580, 74.4001617, -98.7499924, 92.7611694
4: -23.8476677, 34.1320457, -52.8640900, 74.5853195, -98.4329834, 86.9961319

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4817290, upper bound: 96.5432885
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4792915, upper bound: 96.5402277
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4811687, upper bound: 96.5390428
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -18.3205643, 36.8871384, -45.9752731, 78.6173096, -96.9378662, 82.8624039
1: -20.1822853, 30.2803841, -50.2814026, 66.3871689, -86.5694580, 80.5617828
2: -20.7637596, 29.9831238, -51.5448151, 66.4945831, -87.2583466, 81.5279236
3: -24.3498306, 35.0611115, -59.4520378, 76.8625641, -101.2123947, 94.5131378
4: -23.8476677, 34.1320457, -54.4311790, 77.0660400, -100.9137115, 88.5632248

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4902744, upper bound: 96.5856695
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4882219, upper bound: 96.5865405
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4902265, upper bound: 96.5868188
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -41.3976822, 71.2384109, -18.3205643, 36.8871384, -78.2848053, 89.5589676
1: -45.2824707, 60.2273254, -20.1822853, 30.2803841, -75.5628510, 80.4095993
2: -46.4568481, 60.2715683, -20.7637596, 29.9831238, -76.4399643, 81.0353241
3: -53.5106544, 69.7326126, -24.3498306, 35.0611115, -88.5717621, 94.0824432
4: -49.3290863, 69.7293549, -23.8476677, 34.1320457, -83.4611359, 93.5770264

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5235018, upper bound: 96.3886322
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5117359, upper bound: 96.4262611
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5136433, upper bound: 96.4266961
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -41.5644608, 71.5200424, -18.3188400, 36.8851891, -78.4496307, 89.8388748
1: -45.4678535, 60.4032860, -20.1804886, 30.2785072, -75.7463531, 80.5837631
2: -46.6354332, 60.4574471, -20.7618847, 29.9812012, -76.6166306, 81.2193069
3: -53.7209549, 69.9424973, -24.3478432, 35.0588989, -88.7798462, 94.2903366
4: -49.5006561, 69.9765320, -23.8458786, 34.1297989, -83.6304321, 93.8223953

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5367415, upper bound: 96.4671696
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5233055, upper bound: 96.4284680
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5270375, upper bound: 96.4292293
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -41.3976822, 71.2384109, -21.9162598, 42.4145737, -83.8122482, 93.1546707
1: -45.2824707, 60.2273254, -24.0546131, 34.9361496, -80.2186203, 84.2819061
2: -46.4568481, 60.2715683, -24.7162457, 34.6161995, -81.0730438, 84.9878006
3: -53.5106544, 69.7326126, -28.7984715, 40.4117126, -93.9223633, 98.5310669
4: -49.3290863, 69.7293549, -27.7900867, 39.5489616, -88.8780518, 97.5194397

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5342067, upper bound: 96.5049930
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5377097, upper bound: 96.5047957
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5038026, upper bound: 96.4987349
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -41.3431168, 71.1662674, -25.4723930, 47.2078667, -88.5509796, 96.6386566
1: -45.2236176, 60.1638107, -27.9211235, 39.5968437, -84.8204575, 88.0849304
2: -46.3970146, 60.2068901, -28.6655502, 39.3161354, -85.7131500, 88.8724289
3: -53.4439354, 69.6585236, -33.3168831, 45.8657951, -99.3097305, 102.9754028
4: -49.2699509, 69.6511765, -31.8714256, 45.0863838, -94.3563309, 101.5225906

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5279464, upper bound: 96.4880066
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5342520, upper bound: 96.4971075
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5003449, upper bound: 96.4910467
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -41.5644608, 71.5200424, -21.9145050, 42.4126434, -83.9771042, 93.4345322
1: -45.4678535, 60.4032860, -24.0527725, 34.9342651, -80.4021072, 84.4560471
2: -46.6354332, 60.4574471, -24.7143364, 34.6142807, -81.2497101, 85.1717758
3: -53.7209549, 69.9424973, -28.7964439, 40.4094696, -94.1304169, 98.7389374
4: -49.5006561, 69.9765320, -27.7882481, 39.5467072, -89.0473557, 97.7647781

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5470010, upper bound: 96.5069808
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5497951, upper bound: 96.5068810
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5110526, upper bound: 96.4996253
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -41.5103149, 71.4483261, -25.4695473, 47.2052612, -88.7155762, 96.9178619
1: -45.4094810, 60.3402138, -27.9182549, 39.5943451, -85.0038223, 88.2584534
2: -46.5760345, 60.3932266, -28.6625729, 39.3135452, -85.8895798, 89.0557938
3: -53.6548119, 69.8689575, -33.3138962, 45.8628311, -99.5176392, 103.1828461
4: -49.4419518, 69.8989029, -31.8688240, 45.0833549, -94.5253067, 101.7677307

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5407407, upper bound: 96.4899944
time: 1.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5463374, upper bound: 96.4991930
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5075948, upper bound: 96.4919370
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -45.4107056, 77.8863297, -14.9556608, 31.9021549, -77.3128510, 92.8419876
1: -49.6619873, 65.6825562, -16.5861988, 26.0356140, -75.6976013, 82.2687531
2: -50.9203415, 65.7741852, -17.0159492, 25.7218666, -76.6421967, 82.7901230
3: -58.7286339, 76.0346375, -20.1648331, 30.1525993, -88.8812332, 96.1994705
4: -53.7956123, 76.2025909, -20.2020149, 29.1788654, -82.9744644, 96.4046021

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5338451, upper bound: 96.4159159
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5645556, upper bound: 96.4202609
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -45.9752731, 78.6173096, -18.3205643, 36.8871384, -82.8624115, 96.9378586
1: -50.2814026, 66.3871689, -20.1822853, 30.2803841, -80.5617828, 86.5694580
2: -51.5448151, 66.4945831, -20.7637596, 29.9831238, -81.5279236, 87.2583466
3: -59.4520378, 76.8625641, -24.3498306, 35.0611115, -94.5131378, 101.2123947
4: -54.4311790, 77.0660400, -23.8476677, 34.1320457, -88.5632248, 100.9137115

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5606938, upper bound: 96.4861091
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5834003, upper bound: 96.4900028
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -44.7436600, 76.0611954, -21.0142288, 41.1959839, -85.9396439, 97.0754242
1: -48.8998871, 64.2461166, -23.0874786, 33.8037224, -82.7036057, 87.3335953
2: -50.1368523, 64.3910141, -23.7088909, 33.4833374, -83.6201935, 88.0999069
3: -57.7000580, 74.4001617, -27.6857224, 39.0641861, -96.7642365, 102.0858841
4: -52.8640900, 74.5853195, -26.7909698, 38.2138977, -91.0779877, 101.3762894

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5185572, upper bound: 96.5079883
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5185572, upper bound: 96.5182960
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -44.6663589, 75.9108047, -23.3465767, 43.1221008, -87.7884598, 99.2573853
1: -48.8145943, 64.1256561, -25.5338821, 35.9843025, -84.7988968, 89.6595306
2: -50.0497818, 64.2723160, -26.2375641, 35.7268143, -85.7765808, 90.5098724
3: -57.6002083, 74.2610626, -30.3952370, 41.6864510, -99.2866592, 104.6562881
4: -52.7702751, 74.4481812, -29.0679703, 40.9288330, -93.6991119, 103.5161514

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5186007, upper bound: 96.5079883
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5186007, upper bound: 96.5182960
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -45.9752731, 78.6173096, -21.0142288, 41.1959839, -87.1712570, 99.6315308
1: -50.2814026, 66.3871689, -23.0874786, 33.8037224, -84.0851212, 89.4746399
2: -51.5448151, 66.4945831, -23.7088909, 33.4833374, -85.0281525, 90.2034760
3: -59.4520378, 76.8625641, -27.6857224, 39.0641861, -98.5162048, 104.5482712
4: -54.4311790, 77.0660400, -26.7909698, 38.2138977, -92.6450806, 103.8570099

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5973346, upper bound: 96.5238959
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5973346, upper bound: 96.5252675
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -45.8946686, 78.4580688, -23.3465767, 43.1221008, -89.0167694, 101.8046417
1: -50.1919136, 66.2583771, -25.5338821, 35.9843025, -86.1762161, 91.7922440
2: -51.4533615, 66.3675613, -26.2375641, 35.7268143, -87.1801758, 92.6051254
3: -59.3453751, 76.7140198, -30.3952370, 41.6864510, -101.0318069, 107.1092529
4: -54.3331909, 76.9183578, -29.0679703, 40.9288330, -95.2620010, 105.9863281

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5974072, upper bound: 96.5238957
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5974072, upper bound: 96.5252675
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -41.3976822, 71.2384109, -41.3976822, 71.2384109, -112.6360779, 112.6360855
1: -45.2824707, 60.2273254, -45.2824707, 60.2273254, -105.5097733, 105.5097733
2: -46.4568481, 60.2715683, -46.4568481, 60.2715683, -106.7284012, 106.7284012
3: -53.5106544, 69.7326126, -53.5106544, 69.7326126, -123.2432709, 123.2432709
4: -49.3290863, 69.7293549, -49.3290863, 69.7293549, -119.0584412, 119.0584412

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5369726, upper bound: 96.5385528
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5389451, upper bound: 96.5389452
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -41.3976822, 71.2384109, -41.5644608, 71.5200424, -112.9177094, 112.8028717
1: -45.2824707, 60.2273254, -45.4678535, 60.4032860, -105.6857376, 105.6951675
2: -46.4568481, 60.2715683, -46.6354332, 60.4574471, -106.9142761, 106.9069901
3: -53.5106544, 69.7326126, -53.7209549, 69.9424973, -123.4531555, 123.4535675
4: -49.3290863, 69.7293549, -49.5006561, 69.9765320, -119.3056183, 119.2300110

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5369726, upper bound: 96.5472753
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5389451, upper bound: 96.5389452
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -41.5644608, 71.5200424, -41.3976822, 71.2384109, -112.8028717, 112.9177017
1: -45.4678535, 60.4032860, -45.2824707, 60.2273254, -105.6951599, 105.6857376
2: -46.6354332, 60.4574471, -46.4568481, 60.2715683, -106.9069901, 106.9142761
3: -53.7209549, 69.9424973, -53.5106544, 69.7326126, -123.4535599, 123.4531555
4: -49.5006561, 69.9765320, -49.3290863, 69.7293549, -119.2300110, 119.3056183

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5478017, upper bound: 96.5403354
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5515338, upper bound: 96.5410762
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -41.5644608, 71.5200424, -41.5644608, 71.5200424, -113.0845032, 113.0845032
1: -45.4678535, 60.4032860, -45.4678535, 60.4032860, -105.8711166, 105.8711166
2: -46.6354332, 60.4574471, -46.6354332, 60.4574471, -107.0928726, 107.0928726
3: -53.7209549, 69.9424973, -53.7209549, 69.9424973, -123.6634445, 123.6634369
4: -49.5006561, 69.9765320, -49.5006561, 69.9765320, -119.4771729, 119.4771881

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=108.20734405517578
rel_dist={4: [-96.61858128162753, 96.61858128162754]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5299857, upper bound: 96.5919846
time: 0.91 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6152408, upper bound: 96.6152409
time: 0.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 4, lower bound: -96.5299857, upper bound: 96.5919846
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 4, lower bound: -96.6152408, upper bound: 96.6152409

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -23.6491947, 44.5811996, -33.7610512, 56.9028625, -80.5520554, 78.3422546
1: -25.9020424, 36.9293671, -36.8422394, 48.6217384, -74.5237808, 73.7716064
2: -26.6062851, 36.6187668, -37.7350235, 48.5267754, -75.1330566, 74.3537750
3: -30.8861237, 42.7392807, -43.3244705, 56.4820633, -87.3681870, 86.0637512
4: -29.5871983, 41.9198265, -40.1423340, 56.1176567, -85.7048340, 82.0621643

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5180541, upper bound: 96.5180541
time: 0.84 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5180541, upper bound: 96.5919846
time: 0.95 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -47.7588348, 79.9911652, -37.1937485, 61.3177948, -109.0766296, 117.1849060
1: -52.1639366, 67.9121017, -40.5940704, 53.0116119, -105.1755524, 108.5061722
2: -53.4510193, 68.0761642, -41.5752144, 52.9253693, -106.3763885, 109.6513824
3: -61.4133224, 78.7076035, -47.6881599, 61.5470695, -122.9603806, 126.3957520
4: -56.2418251, 79.0030441, -44.0563965, 61.3981972, -117.6400223, 123.0594406

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5919846, upper bound: 96.5299857
time: 0.69 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5919846, upper bound: 96.6152409
time: 1.01 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.94 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.94
Output dim: 4, lower bound: -96.5180541, upper bound: 96.5180541
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.94
Output dim: 4, lower bound: -96.5180541, upper bound: 96.5919846
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.94
Output dim: 4, lower bound: -96.5919846, upper bound: 96.5299857
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.94
Output dim: 4, lower bound: -96.5919846, upper bound: 96.6152409

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -23.6491947, 44.5811996, -47.6231003, 79.8009262, -103.4501190, 92.2042999
1: -25.9020424, 36.9293671, -52.0142975, 67.7256165, -93.6276398, 88.9436646
2: -26.6062851, 36.6187668, -53.2948380, 67.8909683, -94.4972458, 89.9136047
3: -30.8861237, 42.7392807, -61.2372093, 78.4885483, -109.3746719, 103.9764862
4: -29.5871983, 41.9198265, -56.0826988, 78.7833939, -108.3705902, 98.0025253

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4901490, upper bound: 96.5792662
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4901490, upper bound: 96.5060396
time: 0.87 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -47.7588348, 79.9911652, -23.6491947, 44.5811996, -92.3400269, 103.6403580
1: -52.1639366, 67.9121017, -25.9020424, 36.9293671, -89.0933075, 93.8141251
2: -53.4510193, 68.0761642, -26.6062851, 36.6187668, -90.0697784, 94.6824493
3: -61.4133224, 78.7076035, -30.8861237, 42.7392807, -104.1525955, 109.5937271
4: -56.2418251, 79.0030441, -29.5871983, 41.9198265, -98.1616516, 108.5902405

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5504086, upper bound: 96.5137544
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5907111, upper bound: 96.5299703
time: 1.00 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -47.7588348, 79.9911652, -47.7588348, 79.9911652, -127.7499771, 127.7499771
1: -52.1639366, 67.9121017, -52.1639366, 67.9121017, -120.0760345, 120.0760345
2: -53.4510193, 68.0761642, -53.4510193, 68.0761642, -121.5271835, 121.5271835
3: -61.4133224, 78.7076035, -61.4133224, 78.7076035, -140.1209106, 140.1208954
4: -56.2418251, 79.0030441, -56.2418251, 79.0030441, -135.2448730, 135.2448730

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5504089, upper bound: 96.5137545
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5907115, upper bound: 96.5299703
time: 0.97 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.39 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 4, lower bound: -96.4901490, upper bound: 96.5792662
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.39
Output dim: 4, lower bound: -96.4901490, upper bound: 96.5060396
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 4, lower bound: -96.5504086, upper bound: 96.5137544
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 4, lower bound: -96.5907111, upper bound: 96.5299703
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 4, lower bound: -96.5504089, upper bound: 96.5137545
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 4, lower bound: -96.5907115, upper bound: 96.5299703

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -19.0168400, 37.8047943, -45.8653870, 77.6895981, -96.7064362, 83.6701813
1: -20.9385338, 31.1538448, -50.1152267, 65.7206879, -86.6592255, 81.2690735
2: -21.5286846, 30.8517284, -51.3710289, 65.8505325, -87.3792191, 82.2227478
3: -25.2066669, 36.0842285, -59.0704269, 76.1381378, -101.3448029, 95.1546478
4: -24.6514683, 35.1617165, -54.1669464, 76.3358307, -100.9872742, 89.3286591

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4829289, upper bound: 96.5060276
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5010035, upper bound: 96.5790409
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -41.6879997, 71.5514069, -22.2483444, 42.9904251, -84.6784210, 93.7997513
1: -45.5929909, 60.5228043, -24.4221458, 35.4430466, -81.0360413, 84.9449463
2: -46.7689133, 60.5786438, -25.0933800, 35.1113663, -81.8802719, 85.6720200
3: -53.8531685, 70.0829163, -29.2393780, 41.0183983, -94.8715515, 99.3222961
4: -49.6292267, 70.1049042, -28.2274303, 40.1427917, -89.7720184, 98.3323364

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4678775, upper bound: 96.4431286
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5475029, upper bound: 96.5012419
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -47.0677299, 79.0901794, -23.5096550, 44.4128456, -91.4805756, 102.5998383
1: -51.4176712, 67.0940704, -25.7535896, 36.7733688, -88.1910400, 92.8476562
2: -52.6958504, 67.2510605, -26.4548645, 36.4621964, -89.1580429, 93.7059174
3: -60.5642242, 77.7514114, -30.7202072, 42.5562134, -103.1204300, 108.4716187
4: -55.4830666, 78.0209122, -29.4430237, 41.7342072, -97.2172699, 107.4639359

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5276411, upper bound: 96.4637362
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5876024, upper bound: 96.5171388
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -41.6879997, 71.5514069, -45.9063454, 77.7478638, -119.4358673, 117.4577484
1: -45.5929909, 60.5228043, -50.1601906, 65.7792969, -111.3722839, 110.6829910
2: -46.7689133, 60.5786438, -51.4193115, 65.9085159, -112.6774292, 111.9979553
3: -53.8531685, 70.0829163, -59.1220703, 76.2065353, -130.0596924, 129.2049866
4: -49.6292267, 70.1049042, -54.2144699, 76.4023819, -126.0316086, 124.3193741

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5240624, upper bound: 96.4877824
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5638043, upper bound: 96.5863409
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -47.0677299, 79.0901794, -47.6053238, 79.7972412, -126.8649750, 126.6954880
1: -51.4176712, 67.0940704, -51.9992294, 67.7347183, -119.1523895, 119.0932999
2: -52.6958504, 67.2510605, -53.2842026, 67.8974304, -120.5932770, 120.5352554
3: -60.5642242, 77.7514114, -61.2274323, 78.4998932, -139.0641174, 138.9788513
4: -55.4830666, 78.0209122, -56.0743141, 78.7900085, -134.2730713, 134.0952301

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6124160, upper bound: 96.6135347
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6138674, upper bound: 96.6138789
time: 1.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.56 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.56
Output dim: 4, lower bound: -96.4829289, upper bound: 96.5060276
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.56
Output dim: 4, lower bound: -96.5010035, upper bound: 96.5790409
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.56
Output dim: 4, lower bound: -96.4678775, upper bound: 96.4431286
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.56
Output dim: 4, lower bound: -96.5475029, upper bound: 96.5012419
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.56
Output dim: 4, lower bound: -96.5276411, upper bound: 96.4637362
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.56
Output dim: 4, lower bound: -96.5876024, upper bound: 96.5171388
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.56
Output dim: 4, lower bound: -96.5240624, upper bound: 96.4877824
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.56
Output dim: 4, lower bound: -96.5638043, upper bound: 96.5863409
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.56
Output dim: 4, lower bound: -96.6124160, upper bound: 96.6135347
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.56
Output dim: 4, lower bound: -96.6138674, upper bound: 96.6138789

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -18.6725349, 37.3521461, -44.7693863, 77.1989212, -95.8714294, 82.1215363
1: -20.5653000, 30.7257328, -48.9781456, 65.0250778, -85.5903702, 79.7038803
2: -21.1513100, 30.4298763, -50.2283897, 65.0788422, -86.2301407, 80.6582642
3: -24.7898903, 35.5803299, -57.9516258, 75.2778244, -100.0677185, 93.5319519
4: -24.2713909, 34.6660957, -53.1475487, 75.3697815, -99.6411743, 87.8136444

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4648508, upper bound: 96.5185457
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4050241, upper bound: 96.4864863
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4861336, upper bound: 96.5759639
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -41.4855270, 71.2880325, -21.5191193, 42.0334473, -83.5189743, 92.8071518
1: -45.3734360, 60.2761650, -23.6282730, 34.5277176, -79.9011536, 83.9044342
2: -46.5448341, 60.3312607, -24.2926712, 34.1971817, -80.7420120, 84.6239319
3: -53.6018257, 69.7939301, -28.3302307, 39.9395103, -93.5413361, 98.1241608
4: -49.4068031, 69.8075409, -27.3716564, 39.0509071, -88.4577103, 97.1791992

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4687889, upper bound: 96.4436670
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5457880, upper bound: 96.5012419
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4817632, upper bound: 96.4486409
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5339704, upper bound: 96.4928080
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5442989, upper bound: 96.4943909
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -44.4456558, 75.7230988, -18.6471367, 37.9048386, -82.3504944, 94.3702393
1: -48.5413284, 63.8226891, -20.5436020, 31.0511513, -79.5924835, 84.3662720
2: -49.7874222, 63.9342270, -21.1098099, 30.6970921, -80.4844971, 85.0440369
3: -57.2221832, 73.8933792, -24.7835197, 35.9221802, -93.1443634, 98.6768875
4: -52.5007553, 74.0386887, -24.3163395, 34.9955482, -87.4962997, 98.3550186

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5262504, upper bound: 96.4636521
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5127037, upper bound: 96.4523405
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5130071, upper bound: 96.4518687
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -46.8498688, 78.8104706, -22.7081470, 43.4141769, -90.2640457, 101.5186157
1: -51.1817436, 66.8318481, -24.8868942, 35.8187294, -87.0004654, 91.7187347
2: -52.4551163, 66.9864502, -25.5803623, 35.5054359, -87.9605484, 92.5668106
3: -60.2939453, 77.4408493, -29.7423515, 41.4328804, -101.7268219, 107.1831741
4: -55.2434692, 77.7021255, -28.5313034, 40.5888214, -95.8322906, 106.2334290

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5848653, upper bound: 96.5144795
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5867514, upper bound: 96.5164643
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -41.1291542, 70.8217316, -43.6094017, 74.7268448, -115.8560028, 114.4311371
1: -44.9863968, 59.8384171, -47.6706238, 62.9581070, -107.9445038, 107.5090408
2: -46.1558495, 59.8892136, -48.8970566, 63.0581474, -109.2139969, 108.7862701
3: -53.1630325, 69.2790680, -56.2819328, 72.9013519, -126.0643845, 125.5610046
4: -49.0022812, 69.2778473, -51.6553841, 72.9873276, -121.9896088, 120.9332275

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5236964, upper bound: 96.4877824
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5239710, upper bound: 96.4839882
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5072748, upper bound: 96.4864498
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -41.1803513, 70.9833527, -44.7874870, 77.2257767, -118.4061279, 115.7708435
1: -45.0509872, 59.9902573, -48.9980164, 65.0518265, -110.1027985, 108.9882431
2: -46.2223358, 60.0403709, -50.2499580, 65.1054306, -111.3277588, 110.2903137
3: -53.2577057, 69.4607925, -57.9745407, 75.3089600, -128.5666504, 127.4353256
4: -49.0983353, 69.4543762, -53.1687279, 75.3999481, -124.4982834, 122.6231079

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5555636, upper bound: 96.5582049
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5491721, upper bound: 96.5708618
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5630226, upper bound: 96.5854925
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -46.9273148, 78.8993454, -45.3092918, 76.6375046, -123.5648041, 124.2086334
1: -51.2651176, 66.9197693, -49.5031128, 64.8511887, -116.1163025, 116.4228668
2: -52.5400772, 67.0751038, -50.7312202, 64.9858475, -117.5259171, 117.8063202
3: -60.3881569, 77.5475693, -58.3391380, 75.1248856, -135.5130005, 135.8867035
4: -55.3263168, 77.8113403, -53.4937401, 75.3254547, -130.6517639, 131.3050690

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6120710, upper bound: 96.6119746
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6120710, upper bound: 96.6135347
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -46.5381470, 78.0605316, -48.0290642, 79.4345016, -125.9726410, 126.0895920
1: -50.8349686, 66.2686310, -52.4469833, 67.8231888, -118.6581345, 118.7156067
2: -52.0991936, 66.4317398, -53.7335358, 68.0179596, -120.1171417, 120.1652679
3: -59.8780708, 76.7956543, -61.7479401, 78.6598511, -138.5379181, 138.5435944
4: -54.8452110, 77.0760574, -56.4954376, 79.0650940, -133.9103088, 133.5714722

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5724726, upper bound: 96.5885041
time: 1.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6132766, upper bound: 96.6134205
time: 0.94 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.55 seconds
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.4050241, upper bound: 96.4864863
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.4861336, upper bound: 96.5759639
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.5339704, upper bound: 96.4928080
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.5442989, upper bound: 96.4943909
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.5127037, upper bound: 96.4523405
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.5130071, upper bound: 96.4518687
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.5848653, upper bound: 96.5144795
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.5867514, upper bound: 96.5164643
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.5239710, upper bound: 96.4839882
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.5072748, upper bound: 96.4864498
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.5491721, upper bound: 96.5708618
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.5630226, upper bound: 96.5854925
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.6120710, upper bound: 96.6119746
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.6120710, upper bound: 96.6135347
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.5724726, upper bound: 96.5885041
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.55
Output dim: 4, lower bound: -96.6132766, upper bound: 96.6134205

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -17.9892159, 36.4454117, -44.5587158, 76.9236603, -94.9128647, 81.0041122
1: -19.8250408, 29.8654518, -48.7495766, 64.7666016, -84.5916290, 78.6150055
2: -20.3995399, 29.5730305, -49.9939117, 64.8182297, -85.2177734, 79.5669403
3: -23.9467239, 34.5730743, -57.6887131, 74.9736176, -98.9203339, 92.2617874
4: -23.4810486, 33.6490974, -52.9150047, 75.0571594, -98.5381927, 86.5640945

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4757072, upper bound: 96.5396196
time: 1.29 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4757072, upper bound: 96.5759642
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -41.1955223, 70.9753265, -21.5191193, 42.0334473, -83.2289734, 92.4944458
1: -45.0632858, 59.9810600, -23.6282730, 34.5277176, -79.5910034, 83.6093292
2: -46.2331696, 60.0244522, -24.2926712, 34.1971817, -80.4303513, 84.3171234
3: -53.2597275, 69.4442291, -28.3302307, 39.9395103, -93.1992340, 97.7744522
4: -49.1071396, 69.4325027, -27.3716564, 39.0509071, -88.1580505, 96.8041611

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5271161, upper bound: 96.4718181
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5271161, upper bound: 96.4928076
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -41.3621597, 71.2565155, -21.4717484, 41.9835358, -83.3456879, 92.7282562
1: -45.2485199, 60.1567497, -23.5788040, 34.4788322, -79.7273560, 83.7355423
2: -46.4115639, 60.2099724, -24.2415009, 34.1473579, -80.5589218, 84.4514771
3: -53.4697800, 69.6537476, -28.2762184, 39.8812637, -93.3510361, 97.9299622
4: -49.2784882, 69.6792679, -27.3225174, 38.9924812, -88.2709656, 97.0017853

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5388300, upper bound: 96.4737536
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5388300, upper bound: 96.4943909
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -44.5451355, 75.6402054, -22.5703163, 43.2375336, -87.7826691, 98.2105103
1: -48.6766815, 63.9387207, -24.7394638, 35.6561584, -84.3328400, 88.6781845
2: -49.8926659, 64.0645447, -25.4286594, 35.3420105, -85.2346802, 89.4932022
3: -57.3959312, 74.0546494, -29.5753574, 41.2418938, -98.6378174, 103.6300049
4: -52.6537437, 74.2254639, -28.3846893, 40.3955460, -93.0492859, 102.6101532

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5848653, upper bound: 96.5144795
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5848653, upper bound: 96.5144795
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -47.1216125, 78.2919769, -22.3376045, 42.6102409, -89.7318497, 100.6295776
1: -51.4698753, 66.7594910, -24.4754791, 35.2000504, -86.6699219, 91.2349625
2: -52.7412758, 66.9462814, -25.1630363, 34.8971176, -87.6383972, 92.1093140
3: -60.6364861, 77.4105835, -29.2510719, 40.7255669, -101.3620529, 106.6616516
4: -55.4991417, 77.7831726, -28.0742607, 39.8869171, -95.3860626, 105.8574371

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5867514, upper bound: 96.5164643
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5867514, upper bound: 96.5164643
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -40.8695107, 70.6795197, -43.2153244, 74.1633606, -115.0328674, 113.8948135
1: -44.7134666, 59.5705872, -47.2419243, 62.4429436, -107.1563797, 106.8125000
2: -45.8822784, 59.6047897, -48.4577332, 62.5415916, -108.4238739, 108.0625076
3: -52.8981743, 68.9565506, -55.7902794, 72.3023148, -125.2004852, 124.7468262
4: -48.7262993, 68.9838257, -51.2106171, 72.3857346, -121.1120224, 120.1944351

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5226370, upper bound: 96.4784312
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5226043, upper bound: 96.4813512
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5233671, upper bound: 96.4797610
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 40

Time for candidate selection: 11.09 seconds

### Candidate
type: B, layer: 3, pos: 46

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5239277, upper bound: 96.4777257
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5239710, upper bound: 96.4839882
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -40.8864136, 70.6676025, -44.7874870, 77.2257767, -118.1121902, 115.4550781
1: -44.7368355, 59.6921425, -48.9980164, 65.0518265, -109.7886658, 108.6901550
2: -45.9066086, 59.7304153, -50.2499580, 65.1054306, -111.0120239, 109.9803467
3: -52.9115639, 69.1073761, -57.9745407, 75.3089600, -128.2205200, 127.0819168
4: -48.7949791, 69.0757370, -53.1687279, 75.3999481, -124.1949310, 122.2444611

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5412570, upper bound: 96.5481658
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5412570, upper bound: 96.5708618
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -41.0642242, 70.9566193, -44.7244797, 77.1514740, -118.2156906, 115.6810913
1: -44.9334297, 59.8750725, -48.9304123, 64.9808197, -109.9142227, 108.8054810
2: -46.0964050, 59.9235535, -50.1809998, 65.0326614, -111.1290665, 110.1045532
3: -53.1330490, 69.3255539, -57.8982086, 75.2263794, -128.3594055, 127.2237625
4: -48.9763832, 69.3320007, -53.1009560, 75.3135071, -124.2898788, 122.4329376

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544638, upper bound: 96.5544234
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544638, upper bound: 96.5854925
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -44.7764435, 75.9336700, -45.3092918, 76.6375046, -121.4139404, 121.2429657
1: -48.9274368, 64.2144089, -49.5031128, 64.8511887, -113.7786255, 113.7175140
2: -50.1477127, 64.3425446, -50.7312202, 64.9858475, -115.1335602, 115.0737610
3: -57.6820221, 74.3809357, -58.3391380, 75.1248856, -132.8068695, 132.7200623
4: -52.9067955, 74.5601578, -53.4937401, 75.3254547, -128.2322235, 128.0538940

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6023782, upper bound: 96.5245843
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6120710, upper bound: 96.6119746
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -47.3543434, 78.5891724, -45.3092918, 76.6375046, -123.9918365, 123.8984680
1: -51.7220154, 67.0382690, -49.5031128, 64.8511887, -116.5731888, 116.5413666
2: -52.9980049, 67.2279663, -50.7312202, 64.9858475, -117.9838562, 117.9591827
3: -60.9256325, 77.7400436, -58.3391380, 75.1248856, -136.0504913, 136.0791626
4: -55.7538834, 78.1223679, -53.4937401, 75.3254547, -131.0793304, 131.6161041

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6023782, upper bound: 96.5579264
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6120710, upper bound: 96.6119746
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -44.2252884, 75.0488739, -47.4959946, 78.7273941, -122.9526825, 122.5448685
1: -48.3298035, 63.4379005, -51.8661690, 67.1557999, -115.4856033, 115.3040619
2: -49.5531616, 63.5911827, -53.1442719, 67.3549271, -116.9080887, 116.7354584
3: -57.0315056, 73.4633102, -61.0859604, 77.8717194, -134.9032288, 134.5492706
4: -52.2373962, 73.6637878, -55.8835678, 78.2685699, -130.5059357, 129.5473633

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5590466, upper bound: 96.5677323
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5588419, upper bound: 96.5750004
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -45.4626999, 77.5855637, -47.4760017, 78.8283997, -124.2910995, 125.0615540
1: -49.7158012, 65.5603561, -51.8579407, 67.2521210, -116.9679260, 117.4182739
2: -50.9629898, 65.6777649, -53.1384277, 67.4405670, -118.4035568, 118.8161697
3: -58.7792511, 75.9087448, -61.1021881, 77.9916992, -136.7709351, 137.0108948
4: -53.8081703, 76.1230774, -55.9116631, 78.3661575, -132.1743317, 132.0347443

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6075679, upper bound: 96.5917955
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6096781, upper bound: 96.6098807
time: 0.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.70 seconds
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.4757072, upper bound: 96.5396196
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.4757072, upper bound: 96.5759642
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5271161, upper bound: 96.4718181
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5271161, upper bound: 96.4928076
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5388300, upper bound: 96.4737536
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5388300, upper bound: 96.4943909
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5848653, upper bound: 96.5144795
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5848653, upper bound: 96.5144795
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5867514, upper bound: 96.5164643
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5867514, upper bound: 96.5164643
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5239277, upper bound: 96.4777257
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5239710, upper bound: 96.4839882
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5412570, upper bound: 96.5481658
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5412570, upper bound: 96.5708618
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5544638, upper bound: 96.5544234
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5544638, upper bound: 96.5854925
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.6023782, upper bound: 96.5245843
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.6120710, upper bound: 96.6119746
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.6023782, upper bound: 96.5579264
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.6120710, upper bound: 96.6119746
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5590466, upper bound: 96.5677323
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.5588419, upper bound: 96.5750004
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.6075679, upper bound: 96.5917955
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.70
Output dim: 4, lower bound: -96.6096781, upper bound: 96.6098807

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17.9892159, 36.4454117, -40.7963028, 71.2023697, -89.1915817, 77.2416916
1: -19.8250408, 29.8654518, -44.6629257, 59.9626846, -79.7877274, 74.5283508
2: -20.3995399, 29.5730305, -45.8396568, 59.9544525, -80.3539886, 75.4126816
3: -23.9467239, 34.5730743, -52.9413948, 69.3893280, -93.3360519, 87.5144653
4: -23.4810486, 33.6490974, -48.8139801, 69.3144913, -92.7955246, 82.4630737

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4733969, upper bound: 96.5380685
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4757072, upper bound: 96.5394867
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.9892159, 36.4454117, -45.5758209, 78.1139374, -96.1031342, 82.0212250
1: -19.8250408, 29.8654518, -49.8496857, 65.9114075, -85.7364502, 79.7151184
2: -20.3995399, 29.5730305, -51.1028404, 66.0127411, -86.4122696, 80.6758575
3: -23.9467239, 34.5730743, -58.9599419, 76.3022614, -100.2489853, 93.5330200
4: -23.4810486, 33.6490974, -53.9938812, 76.4885406, -99.9695740, 87.6429749

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4733969, upper bound: 96.5731573
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4757072, upper bound: 96.5758294
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -41.1955223, 70.9753265, -18.3205643, 36.8871384, -78.0826569, 89.2958908
1: -45.0632858, 59.9810600, -20.1822853, 30.2803841, -75.3436737, 80.1633453
2: -46.2331696, 60.0244522, -20.7637596, 29.9831238, -76.2162933, 80.7882080
3: -53.2597275, 69.4442291, -24.3498306, 35.0611115, -88.3208237, 93.7940598
4: -49.1071396, 69.4325027, -23.8476677, 34.1320457, -83.2391815, 93.2801666

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5223239, upper bound: 96.4641837
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5271150, upper bound: 96.4718181
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4850640, upper bound: 96.4644046
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -41.1955223, 70.9753265, -22.1539459, 42.7442818, -83.9398041, 93.1292648
1: -45.0632858, 59.9810600, -24.2990932, 35.2033768, -80.2666626, 84.2801514
2: -46.2331696, 60.0244522, -24.9784660, 34.8872604, -81.1204224, 85.0029144
3: -53.2597275, 69.4442291, -29.0852947, 40.7103271, -93.9700470, 98.5295181
4: -49.1071396, 69.4325027, -27.9635525, 39.8563309, -88.9634705, 97.3960571

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5223239, upper bound: 96.4928080
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5271150, upper bound: 96.4918251
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4850640, upper bound: 96.4843446
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -41.3621597, 71.2565155, -18.2727108, 36.8332100, -78.1953659, 89.5292282
1: -45.2485199, 60.1567497, -20.1324272, 30.2284927, -75.4770126, 80.2891769
2: -46.4115639, 60.2099724, -20.7117519, 29.9299870, -76.3415527, 80.9217072
3: -53.4697800, 69.6537476, -24.2947502, 34.9999199, -88.4696884, 93.9484940
4: -49.2784882, 69.6792679, -23.7981319, 34.0701408, -83.3486176, 93.4773941

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5305093, upper bound: 96.4654329
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5388287, upper bound: 96.4737536
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4942864, upper bound: 96.4653807
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -41.3621597, 71.2565155, -22.1071548, 42.6948166, -84.0569611, 93.3636475
1: -45.2485199, 60.1567497, -24.2502365, 35.1549377, -80.4034576, 84.4069824
2: -46.4115639, 60.2099724, -24.9279060, 34.8378639, -81.2494278, 85.1378708
3: -53.4697800, 69.6537476, -29.0318127, 40.6530228, -94.1227951, 98.6855621
4: -49.2784882, 69.6792679, -27.9150620, 39.7983932, -89.0768738, 97.5943222

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5305093, upper bound: 96.4943909
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5388287, upper bound: 96.4934032
time: 6.32 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4942864, upper bound: 96.4853575
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -44.5451355, 75.6402054, -20.7375793, 40.8051033, -85.3502350, 96.3777771
1: -48.6766815, 63.9387207, -22.7748871, 33.4115906, -82.0882721, 86.7136078
2: -49.8926659, 64.0645447, -23.4025745, 33.0925522, -82.9852142, 87.4671173
3: -57.3959312, 74.0546494, -27.3185463, 38.6041641, -96.0000916, 101.3731842
4: -52.6537437, 74.2254639, -26.4089775, 37.7383881, -90.3921356, 100.6344299

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5217569, upper bound: 96.4623158
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5059161, upper bound: 96.4891426
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5821417, upper bound: 96.5043626
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -44.5451355, 75.6402054, -22.9815292, 42.6599121, -87.2050476, 98.6217346
1: -48.6766815, 63.9387207, -25.1322250, 35.5264778, -84.2031555, 89.0709381
2: -49.8926659, 64.0645447, -25.8413391, 35.2649612, -85.1576233, 89.9058838
3: -57.3959312, 74.0546494, -29.9394932, 41.1514778, -98.5473938, 103.9941330
4: -52.6537437, 74.2254639, -28.6162453, 40.3743286, -93.0280533, 102.8416977

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5217569, upper bound: 96.4623158
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5059161, upper bound: 96.4891426
time: 2.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5821417, upper bound: 96.5043626
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -47.1216125, 78.2919769, -20.7375793, 40.8051033, -87.9267120, 99.0295563
1: -51.4698753, 66.7594910, -22.7748871, 33.4115906, -84.8814697, 89.5343781
2: -52.7412758, 66.9462814, -23.4025745, 33.0925522, -85.8338242, 90.3488541
3: -60.6364861, 77.4105835, -27.3185463, 38.6041641, -99.2406464, 104.7291260
4: -55.4991417, 77.7831726, -26.4089775, 37.7383881, -93.2375336, 104.1921387

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5545281, upper bound: 96.5007281
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5830552, upper bound: 96.5062493
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -47.1216125, 78.2919769, -22.9815292, 42.6599121, -89.7815247, 101.2735062
1: -51.4698753, 66.7594910, -25.1322250, 35.5264778, -86.9963531, 91.8917007
2: -52.7412758, 66.9462814, -25.8413391, 35.2649612, -88.0062408, 92.7876205
3: -60.6364861, 77.4105835, -29.9394932, 41.1514778, -101.7879639, 107.3500748
4: -55.4991417, 77.7831726, -28.6162453, 40.3743286, -95.8734665, 106.3994141

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5545281, upper bound: 96.5007284
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5830552, upper bound: 96.5062493
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -40.8695107, 70.6795197, -39.1177139, 67.4608765, -108.3303757, 109.7972107
1: -44.7134666, 59.5705872, -42.7554207, 56.1009369, -100.8143997, 102.3259964
2: -45.8822784, 59.6047897, -43.8082199, 56.2489166, -102.1311798, 103.4130096
3: -52.8981743, 68.9565506, -50.3650894, 65.0292892, -117.9274597, 119.3216400
4: -48.7262993, 68.9838257, -46.2903519, 65.2711105, -113.9973679, 115.2741776

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5226896, upper bound: 96.4775967
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40

Time for candidate selection: 8.31 seconds

### Candidate
type: A, layer: 3, pos: 46

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5211293, upper bound: 96.4772560
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5211293, upper bound: 96.4777257
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -40.8695107, 70.6795197, -40.5668259, 69.7702560, -110.6397705, 111.2463379
1: -44.7134666, 59.5705872, -44.3728333, 59.3714790, -104.0849304, 103.9434204
2: -45.8822784, 59.6047897, -45.5656281, 59.4113464, -105.2936172, 105.1704178
3: -52.8981743, 68.9565506, -52.5436630, 68.7512894, -121.6494522, 121.5002136
4: -48.7262993, 68.9838257, -48.3538361, 68.4993057, -117.2255630, 117.3376617

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5236964, upper bound: 96.4839882
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40

Time for candidate selection: 8.69 seconds

### Candidate
type: A, layer: 3, pos: 46

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5211293, upper bound: 96.4834666
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5211293, upper bound: 96.4839882
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -40.8864136, 70.6676025, -40.9972687, 71.4684448, -112.3548584, 111.6648560
1: -44.7368355, 59.6921425, -44.8812256, 60.2124519, -104.9492874, 104.5733643
2: -45.9066086, 59.7304153, -46.0632477, 60.2060394, -106.1126404, 105.7936630
3: -52.9115639, 69.1073761, -53.1930847, 69.6801071, -122.5916595, 122.3004608
4: -48.7949791, 69.0757370, -49.0351067, 69.6159058, -118.4108887, 118.1108398

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5379393, upper bound: 96.5471462
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5398866, upper bound: 96.5474299
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -40.8864136, 70.6676025, -45.8210030, 78.4321976, -119.3186111, 116.4886017
1: -44.7368355, 59.6921425, -50.1154366, 66.2140503, -110.9508820, 109.8075714
2: -45.9066086, 59.7304153, -51.3768768, 66.3165512, -112.2231598, 111.1072693
3: -52.9115639, 69.1073761, -59.2627907, 76.6592026, -129.5707550, 128.3701477
4: -48.7949791, 69.0757370, -54.2646713, 76.8505554, -125.6455383, 123.3404083

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5379393, upper bound: 96.5659635
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5398866, upper bound: 96.5659187
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -41.0642242, 70.9566193, -40.9377594, 71.3968811, -112.4611053, 111.8943710
1: -44.9334297, 59.8750725, -44.8174019, 60.1435318, -105.0769577, 104.6924744
2: -46.0964050, 59.9235535, -45.9977150, 60.1360283, -106.2324371, 105.9212646
3: -53.1330490, 69.3255539, -53.1208954, 69.6002502, -122.7332840, 122.4464493
4: -48.9763832, 69.3320007, -48.9700012, 69.5327225, -118.5090942, 118.3020020

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5404085, upper bound: 96.5530775
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544638, upper bound: 96.5544234
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -41.0642242, 70.9566193, -45.7516327, 78.3515015, -119.4157028, 116.7082520
1: -44.9334297, 59.8750725, -50.0409927, 66.1371155, -111.0705414, 109.9160614
2: -46.0964050, 59.9235535, -51.3011780, 66.2375870, -112.3339920, 111.2247238
3: -53.1330490, 69.3255539, -59.1792793, 76.5696869, -129.7027283, 128.5048065
4: -48.9763832, 69.3320007, -54.1902008, 76.7564468, -125.7328339, 123.5222015

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5404085, upper bound: 96.5751157
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544638, upper bound: 96.5768065
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -44.3936348, 75.4121933, -43.5565338, 74.2641296, -118.6577606, 118.9687271
1: -48.5120087, 63.7150192, -47.6020775, 62.5951157, -111.1071243, 111.3170929
2: -49.7192535, 63.8445702, -48.7808990, 62.7087402, -112.4279938, 112.6254654
3: -57.2017441, 73.7899933, -56.1387787, 72.4747772, -129.6765137, 129.9287720
4: -52.4568329, 73.9645081, -51.4923515, 72.6052017, -125.0620270, 125.4568634

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5140010, upper bound: 96.5319816
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6010406, upper bound: 96.5533918
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -44.4614868, 75.5288696, -47.6931152, 80.3296509, -124.7911377, 123.2219696
1: -48.5889816, 63.8568420, -52.0833817, 68.2822647, -116.8712463, 115.9402237
2: -49.8028488, 63.9794884, -53.4315109, 68.3860779, -118.1889267, 117.4109955
3: -57.3013077, 73.9619446, -61.3820419, 79.0090790, -136.3103485, 135.3439941
4: -52.5656929, 74.1219406, -56.3014832, 79.2456055, -131.8112946, 130.4234009

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5136977, upper bound: 96.5325067
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6116949, upper bound: 96.6116595
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -46.9764595, 78.0713120, -43.5565338, 74.2641296, -121.2405853, 121.6278458
1: -51.3108101, 66.5413513, -47.6020775, 62.5951157, -113.9059296, 114.1434250
2: -52.5750084, 66.7342987, -48.7808990, 62.7087402, -115.2837524, 115.5151978
3: -60.4499626, 77.1527252, -56.1387787, 72.4747772, -132.9247284, 133.2914734
4: -55.3100166, 77.5308304, -51.4923515, 72.6052017, -127.9152222, 129.0231628

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5619908, upper bound: 96.5475433
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6019531, upper bound: 96.5548003
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -47.0119438, 78.1461563, -47.6931152, 80.3296509, -127.3415985, 125.8392487
1: -51.3534241, 66.6472092, -52.0833817, 68.2822647, -119.6356812, 118.7305908
2: -52.6231461, 66.8305740, -53.4315109, 68.3860779, -121.0092239, 120.2620697
3: -60.5099068, 77.2835999, -61.3820419, 79.0090790, -139.5189819, 138.6656189
4: -55.3834991, 77.6420441, -56.3014832, 79.2456055, -134.6291046, 133.9434967

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5892372, upper bound: 96.6027465
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5682027, upper bound: 96.5876127
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6120401, upper bound: 96.6130057
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -43.6541214, 74.2796555, -46.9354324, 78.0825424, -121.7366638, 121.2150879
1: -47.7191086, 62.7372780, -51.2850304, 66.7457657, -114.4648743, 114.0223083
2: -48.9226608, 62.8889008, -52.5272903, 66.9338531, -115.8565140, 115.4161911
3: -56.3315659, 72.6435089, -60.4355087, 77.3929138, -133.7244873, 133.0790100
4: -51.5856781, 72.8241730, -55.2721939, 77.7238998, -129.3095703, 128.0963745

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5375404, upper bound: 96.5416720
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5560648, upper bound: 96.5647324
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -43.8965569, 74.6091461, -45.8295746, 76.5121689, -120.4087067, 120.4387207
1: -47.9800568, 63.0457611, -50.0929375, 65.1907196, -113.1707611, 113.1387024
2: -49.1926460, 63.1942520, -51.3278275, 65.3509521, -114.5435944, 114.5220795
3: -56.6347694, 73.0116196, -59.0779190, 75.6209030, -132.2556763, 132.0895386
4: -51.8741455, 73.1943512, -54.0729027, 75.9095612, -127.7837067, 127.2672577

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5227357, upper bound: 96.5335787
time: 1.23 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5227357, upper bound: 96.5750005
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -44.9361572, 76.8591843, -46.9539261, 78.2298965, -123.1660309, 123.8130951
1: -49.1520081, 64.9118729, -51.3179512, 66.8924942, -116.0444794, 116.2298279
2: -50.3811607, 65.0274658, -52.5632477, 67.0668793, -117.4480286, 117.5907059
3: -58.1346512, 75.1475983, -60.4990959, 77.5669708, -135.7016296, 135.6466980
4: -53.2058868, 75.3416672, -55.3435860, 77.8783493, -131.0842285, 130.6852264

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5460242, upper bound: 96.5461858
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5945971, upper bound: 96.5850324
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5948873, upper bound: 96.5857806
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -45.0876846, 77.1013870, -45.9174232, 76.6827621, -121.7704315, 123.0188141
1: -49.3169823, 65.1250916, -50.1936798, 65.3560104, -114.6729889, 115.3187714
2: -50.5537834, 65.2372589, -51.4317741, 65.5088730, -116.0626526, 116.6690369
3: -58.3319244, 75.4064865, -59.2026215, 75.8215408, -134.1534729, 134.6091003
4: -53.3985291, 75.6005173, -54.2014732, 76.0964432, -129.4949646, 129.8019562

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6085980, upper bound: 96.6088354
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6085980, upper bound: 96.6086093
time: 0.93 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.66 seconds
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.4733969, upper bound: 96.5380685
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.4757072, upper bound: 96.5394867
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.4733969, upper bound: 96.5731573
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.4757072, upper bound: 96.5758294
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5271150, upper bound: 96.4718181
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.4850640, upper bound: 96.4644046
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5271150, upper bound: 96.4918251
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.4850640, upper bound: 96.4843446
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5388287, upper bound: 96.4737536
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.4942864, upper bound: 96.4653807
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5388287, upper bound: 96.4934032
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.4942864, upper bound: 96.4853575
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5059161, upper bound: 96.4891426
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5821417, upper bound: 96.5043626
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5059161, upper bound: 96.4891426
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5821417, upper bound: 96.5043626
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5545281, upper bound: 96.5007281
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5830552, upper bound: 96.5062493
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5545281, upper bound: 96.5007284
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5830552, upper bound: 96.5062493
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5211293, upper bound: 96.4772560
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5211293, upper bound: 96.4777257
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5211293, upper bound: 96.4834666
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5211293, upper bound: 96.4839882
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5379393, upper bound: 96.5471462
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5398866, upper bound: 96.5474299
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5379393, upper bound: 96.5659635
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5398866, upper bound: 96.5659187
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5404085, upper bound: 96.5530775
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5544638, upper bound: 96.5544234
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5404085, upper bound: 96.5751157
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5544638, upper bound: 96.5768065
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5140010, upper bound: 96.5319816
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.6010406, upper bound: 96.5533918
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5136977, upper bound: 96.5325067
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.6116949, upper bound: 96.6116595
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5619908, upper bound: 96.5475433
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.6019531, upper bound: 96.5548003
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5682027, upper bound: 96.5876127
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.6120401, upper bound: 96.6130057
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5375404, upper bound: 96.5416720
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5560648, upper bound: 96.5647324
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5227357, upper bound: 96.5335787
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5227357, upper bound: 96.5750005
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5945971, upper bound: 96.5850324
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.5948873, upper bound: 96.5857806
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.6085980, upper bound: 96.6088354
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.66
Output dim: 4, lower bound: -96.6085980, upper bound: 96.6086093

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -17.1276894, 35.2840424, -40.4160080, 70.6844025, -87.8120880, 75.7000427
1: -18.9065285, 28.8044472, -44.2488556, 59.4714432, -78.3779678, 73.0532913
2: -19.4497108, 28.5143585, -45.4184952, 59.4586487, -78.9083405, 73.9328537
3: -22.8843861, 33.3371582, -52.4585800, 68.8140488, -91.6984329, 85.7957153
4: -22.5671005, 32.4002228, -48.3808022, 68.7224579, -91.2895584, 80.7810211

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4656468, upper bound: 96.5312641
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4688024, upper bound: 96.5263567
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4704473, upper bound: 96.5369985
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.7639732, 40.1978149, -40.6021690, 70.9263840, -91.6903534, 80.7999878
1: -22.8420525, 33.5697899, -44.4521866, 59.7237740, -82.5658188, 78.0219727
2: -23.5049286, 33.3005753, -45.6250343, 59.7120857, -83.2170105, 78.9255905
3: -27.4963703, 38.8959808, -52.6995544, 69.1110382, -96.6074066, 91.5955353
4: -26.7366276, 38.0280914, -48.6008148, 69.0223312, -95.7589569, 86.6288910

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4710370, upper bound: 96.5267101
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4729747, upper bound: 96.5384258
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17.1276894, 35.2840424, -45.1958351, 77.5988846, -94.7265625, 80.4798737
1: -18.9065285, 28.8044472, -49.4379387, 65.4248428, -84.3313675, 78.2423782
2: -19.4497108, 28.5143585, -50.6815720, 65.5181274, -84.9678345, 79.1959229
3: -22.8843861, 33.3371582, -58.4816551, 75.7322769, -98.6166611, 91.8187866
4: -22.5671005, 32.4002228, -53.5649757, 75.8992767, -98.4663773, 85.9651794

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4754914, upper bound: 96.5624508
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4745683, upper bound: 96.5577205
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.7639732, 40.1978149, -45.4003296, 77.8581543, -98.6221237, 85.5981445
1: -22.8420525, 33.5697899, -49.6590538, 65.6945190, -88.5365601, 83.2288437
2: -23.5049286, 33.3005753, -50.9090042, 65.7913361, -89.2962646, 84.2095642
3: -27.4963703, 38.8959808, -58.7409172, 76.0495071, -103.5458755, 97.6369019
4: -26.7366276, 38.0280914, -53.7984543, 76.2227936, -102.9593964, 91.8265305

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4760374, upper bound: 96.5143594
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4760374, upper bound: 96.5758298
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -41.0307312, 70.9450607, -17.9178047, 36.2776184, -77.3083496, 88.8628540
1: -44.8909225, 59.8154030, -19.7462234, 29.7167377, -74.6076584, 79.5616302
2: -46.0631790, 59.8447227, -20.3119144, 29.4232159, -75.4863892, 80.1566238
3: -53.1075630, 69.2419281, -23.8447266, 34.3960342, -87.5035934, 93.0866394
4: -48.9315071, 69.2642365, -23.3649464, 33.4887733, -82.4202728, 92.6291809

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5267958, upper bound: 96.4701170
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5271066, upper bound: 96.4718181
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -41.0307312, 70.9450607, -21.7631454, 42.1475449, -83.1782761, 92.7081985
1: -44.8909225, 59.8154030, -23.8728199, 34.6510315, -79.5419540, 83.6882248
2: -46.0631790, 59.8447227, -24.5405197, 34.3386307, -80.4018097, 84.3852386
3: -53.1075630, 69.2419281, -28.5921936, 40.0616226, -93.1691742, 97.8341064
4: -48.9315071, 69.2642365, -27.4911442, 39.2261925, -88.1576996, 96.7553787

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5338198, upper bound: 96.4899955
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5299241, upper bound: 96.4821995
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -41.1779480, 71.2062073, -17.8701019, 36.2238312, -77.4017792, 89.0763016
1: -45.0547981, 59.9706955, -19.6965752, 29.6650372, -74.7198334, 79.6672516
2: -46.2203102, 60.0092697, -20.2601318, 29.3702736, -75.5905838, 80.2693939
3: -53.2932816, 69.4268494, -23.7898731, 34.3350830, -87.6283646, 93.2167206
4: -49.0833206, 69.4846649, -23.3156471, 33.4270782, -82.5103912, 92.8002930

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5374347, upper bound: 96.4717639
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5388205, upper bound: 96.4737536
time: 1.00 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.83 seconds
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.4688024, upper bound: 96.5263567
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.4704473, upper bound: 96.5369985
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.4710370, upper bound: 96.5267101
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.4729747, upper bound: 96.5384258
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.4754914, upper bound: 96.5624508
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.4745683, upper bound: 96.5577205
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.4760374, upper bound: 96.5143594
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.4760374, upper bound: 96.5758298
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.5267958, upper bound: 96.4701170
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.5271066, upper bound: 96.4718181
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.5338198, upper bound: 96.4899955
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.5299241, upper bound: 96.4821995
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.5374347, upper bound: 96.4717639
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.83
Output dim: 4, lower bound: -96.5388205, upper bound: 96.4737536
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5388287, upper bound: 96.4934032
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5821417, upper bound: 96.5043626
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5821417, upper bound: 96.5043626
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5545281, upper bound: 96.5007281
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5830552, upper bound: 96.5062493
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5545281, upper bound: 96.5007284
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5830552, upper bound: 96.5062493
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5379393, upper bound: 96.5471462
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5398866, upper bound: 96.5474299
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5379393, upper bound: 96.5659635
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5398866, upper bound: 96.5659187
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5404085, upper bound: 96.5530775
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5544638, upper bound: 96.5544234
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5404085, upper bound: 96.5751157
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5544638, upper bound: 96.5768065
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5140010, upper bound: 96.5319816
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.6010406, upper bound: 96.5533918
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5136977, upper bound: 96.5325067
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.6116949, upper bound: 96.6116595
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5619908, upper bound: 96.5475433
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.6019531, upper bound: 96.5548003
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5682027, upper bound: 96.5876127
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.6120401, upper bound: 96.6130057
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5375404, upper bound: 96.5416720
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5560648, upper bound: 96.5647324
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5227357, upper bound: 96.5335787
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5227357, upper bound: 96.5750005
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5945971, upper bound: 96.5850324
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.5948873, upper bound: 96.5857806
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.6085980, upper bound: 96.6088354
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 4, lower bound: -96.6085980, upper bound: 96.6086093
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=108.20734405517578
rel_dist={4: [-96.61854736505002, 96.61854736505003]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5250594, upper bound: 96.5622786
time: 0.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6149788, upper bound: 96.6149788
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 4, lower bound: -96.5250594, upper bound: 96.5622786
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 4, lower bound: -96.6149788, upper bound: 96.6149788

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -23.6491947, 44.5811996, -28.6279411, 50.2904282, -73.9396210, 73.2091370
1: -25.9020424, 36.9293671, -31.2576790, 42.2871094, -68.1891479, 68.1870422
2: -26.6062851, 36.6187668, -32.0478058, 42.0906754, -68.6969604, 68.6665649
3: -30.8861237, 42.7392807, -36.8740997, 49.0579185, -79.9440460, 79.6133804
4: -29.5871983, 41.9198265, -34.5087357, 48.4171104, -78.0043106, 76.4285583

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5178951, upper bound: 96.5178951
time: 0.86 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5178951, upper bound: 96.5622786
time: 0.81 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -47.7588348, 79.9911652, -36.4958420, 60.3437691, -108.1026001, 116.4869995
1: -52.1639366, 67.9121017, -39.8354340, 52.1126213, -104.2765503, 107.7475357
2: -53.4510193, 68.0761642, -40.8007240, 52.0339279, -105.4849472, 108.8768921
3: -61.4133224, 78.7076035, -46.8292847, 60.4491501, -121.8624649, 125.5368805
4: -56.2418251, 79.0030441, -43.2310524, 60.3504219, -116.5922470, 122.2340851

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6047849, upper bound: 96.6044294
time: 0.86 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6056928, upper bound: 96.6056929
time: 1.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.05 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 4.05
Output dim: 4, lower bound: -96.5178951, upper bound: 96.5178951
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 4, lower bound: -96.5178951, upper bound: 96.5622786
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 4, lower bound: -96.6047849, upper bound: 96.6044294
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 4, lower bound: -96.6056928, upper bound: 96.6056929

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -23.6491947, 44.5811996, -46.9844093, 78.7641678, -102.4133606, 91.5655975
1: -25.9020424, 36.9293671, -51.3043747, 66.8036575, -92.7056808, 88.2337418
2: -26.6062851, 36.6187668, -52.5692329, 66.9609375, -93.5672226, 89.1879959
3: -30.8861237, 42.7392807, -60.3899193, 77.4137115, -108.2998276, 103.1291962
4: -29.5871983, 41.9198265, -55.3163528, 77.6945419, -107.2817383, 97.2361755

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4869631, upper bound: 96.5458173
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5178417, upper bound: 96.5610185
time: 1.01 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -44.2398682, 74.9095993, -29.8533764, 50.4144211, -94.6542892, 104.7629776
1: -48.3615913, 63.4710732, -32.6329384, 43.3986778, -91.7602615, 96.1039886
2: -49.5675240, 63.5649490, -33.4083748, 43.2522278, -92.8197479, 96.9733276
3: -57.0275726, 73.5777435, -38.4902649, 50.2862930, -107.3138657, 112.0680084
4: -52.3435822, 73.7183151, -35.7085190, 50.0480194, -102.3916016, 109.4268341

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5980335, upper bound: 96.6001234
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6034937, upper bound: 96.6027402
time: 1.06 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -46.6714134, 78.5828323, -35.1331100, 58.4906158, -105.1620102, 113.7159271
1: -50.9978943, 66.6106262, -38.3722992, 50.4447517, -101.4426422, 104.9829254
2: -52.2598877, 66.7749863, -39.3082428, 50.3677979, -102.6276855, 106.0832214
3: -60.1004333, 77.1704712, -45.1748734, 58.5045624, -118.6049728, 122.3453293
4: -54.9896698, 77.4624481, -41.6677170, 58.3766212, -113.3662872, 119.1301651

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5988484, upper bound: 96.6018719
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6045424, upper bound: 96.6045425
time: 0.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.49 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.49
Output dim: 4, lower bound: -96.4869631, upper bound: 96.5458173
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.49
Output dim: 4, lower bound: -96.5178417, upper bound: 96.5610185
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.49
Output dim: 4, lower bound: -96.5980335, upper bound: 96.6001234
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.49
Output dim: 4, lower bound: -96.6034937, upper bound: 96.6027402
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.49
Output dim: 4, lower bound: -96.5988484, upper bound: 96.6018719
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.49
Output dim: 4, lower bound: -96.6045424, upper bound: 96.6045425

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -19.0168400, 37.8047943, -43.9747543, 75.1367416, -94.1535797, 81.7795486
1: -20.9385338, 31.1538448, -48.0613899, 63.3843040, -84.3228378, 79.2152328
2: -21.5286846, 30.8517284, -49.2877769, 63.4590683, -84.9877472, 80.1395035
3: -25.2066669, 36.0842285, -56.6864853, 73.4260483, -98.6327133, 92.7707062
4: -24.6514683, 35.1617165, -52.0896606, 73.5081177, -98.1595688, 87.2513733

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4867917, upper bound: 96.5084393
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4951251, upper bound: 96.5457661
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.9655075, 43.7550316, -46.6314545, 78.3108902, -101.2763977, 90.3864822
1: -25.1754646, 36.1651459, -50.9241104, 66.3914871, -91.5669479, 87.0892487
2: -25.8641987, 35.8516693, -52.1842804, 66.5449677, -92.4091644, 88.0359497
3: -30.0730324, 41.8423233, -59.9584427, 76.9312057, -107.0042419, 101.8007660
4: -28.8824005, 41.0101280, -54.9295425, 77.1983490, -106.0807343, 95.9396667

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5238321, upper bound: 96.5590107
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5250360, upper bound: 96.5598081
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -42.3073769, 72.1774979, -29.1791306, 49.4326935, -91.7400589, 101.3566284
1: -46.2540817, 60.9855385, -31.8924179, 42.5130234, -88.7671051, 92.8779602
2: -47.4086990, 61.0578918, -32.6545944, 42.3652763, -89.7739716, 93.7124863
3: -54.5746841, 70.6597672, -37.6260147, 49.2638130, -103.8385010, 108.2857742
4: -50.1417389, 70.7389145, -34.9391403, 48.9984741, -99.1402054, 105.6780319

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5964113, upper bound: 96.5976979
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5965256, upper bound: 96.5987456
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -44.7056580, 74.5382843, -29.1164627, 48.9216003, -93.6272583, 103.6547470
1: -48.8465652, 63.5449524, -31.8253479, 42.2213936, -91.0679550, 95.3702927
2: -50.0575027, 63.6693344, -32.5812035, 42.0893364, -92.1468353, 96.2505341
3: -57.5736580, 73.7311249, -37.5502586, 48.9334259, -106.5070801, 111.2813873
4: -52.8008232, 73.9833755, -34.8258057, 48.7041550, -101.5049591, 108.8091736

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6020460, upper bound: 96.6000347
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6029496, upper bound: 96.6019702
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6029496, upper bound: 96.6027402
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -44.4433289, 75.4934387, -34.1216431, 57.0998993, -101.5432281, 109.6150818
1: -48.5721359, 63.7915649, -37.2699203, 49.1719246, -97.7440414, 101.0614853
2: -49.7805099, 63.9304848, -38.1863174, 49.0858994, -98.8663940, 102.1168060
3: -57.2882996, 73.8767395, -43.8982315, 57.0168343, -114.3051300, 117.7749481
4: -52.4813309, 74.0782013, -40.5338440, 56.8570633, -109.3383865, 114.6120453

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5596206, upper bound: 96.5238985
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5596211, upper bound: 96.5981848
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -47.3946037, 78.5754395, -34.4866486, 57.0905838, -104.4851837, 113.0620880
1: -51.7649879, 67.0443039, -37.6568527, 49.3568764, -101.1218414, 104.7011566
2: -53.0343628, 67.2455750, -38.5730133, 49.2934723, -102.3278351, 105.8185806
3: -60.9779358, 77.7328720, -44.3222046, 57.2439919, -118.2219238, 122.0550613
4: -55.7437134, 78.1421356, -40.8740845, 57.1314545, -112.8751602, 119.0162201

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5952739, upper bound: 96.5869278
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5969090, upper bound: 96.5969091
time: 1.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.46 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.46
Output dim: 4, lower bound: -96.4867917, upper bound: 96.5084393
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 4, lower bound: -96.4951251, upper bound: 96.5457661
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 4, lower bound: -96.5238321, upper bound: 96.5590107
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 4, lower bound: -96.5250360, upper bound: 96.5598081
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 4, lower bound: -96.5964113, upper bound: 96.5976979
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 4, lower bound: -96.5965256, upper bound: 96.5987456
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 4, lower bound: -96.6029496, upper bound: 96.6019702
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 4, lower bound: -96.6029496, upper bound: 96.6027402
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 4, lower bound: -96.5596206, upper bound: 96.5238985
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 4, lower bound: -96.5596211, upper bound: 96.5981848
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 4, lower bound: -96.5952739, upper bound: 96.5869278
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 4, lower bound: -96.5969090, upper bound: 96.5969091

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -18.5909042, 37.1508255, -46.3140182, 78.5703049, -97.1612091, 83.4648438
1: -20.4784966, 30.5979939, -50.5923157, 66.6613846, -87.1398773, 81.1903076
2: -21.0562057, 30.2992496, -51.9370308, 66.7077026, -87.7639084, 82.2362823
3: -24.6884918, 35.4343071, -59.6442108, 77.1572189, -101.8457108, 95.0785141
4: -24.1704063, 34.5107269, -54.8679047, 77.2840500, -101.4544525, 89.3786316

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4760176, upper bound: 96.5438975
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4927647, upper bound: 96.5444218
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.0010643, 42.5241013, -44.3970337, 75.3124924, -97.3135376, 86.9211273
1: -24.1462631, 35.0259628, -48.4995232, 63.6381607, -87.7844162, 83.5254822
2: -24.8023777, 34.7061081, -49.7060699, 63.7604141, -88.5627670, 84.4121780
3: -28.9048576, 40.5030441, -57.1624527, 73.7039413, -102.6087952, 97.6654816
4: -27.8582439, 39.6564217, -52.4288597, 73.8736267, -101.7318649, 92.0852585

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5238321, upper bound: 96.5590107
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5238321, upper bound: 96.5590107
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.2751484, 42.2657852, -46.6879501, 77.4099503, -99.6850891, 88.9537201
1: -24.4092617, 35.0159683, -50.9693413, 66.0052872, -90.4145355, 85.9852982
2: -25.0865440, 34.7208862, -52.2223511, 66.1905823, -91.2771301, 86.9432373
3: -29.1567383, 40.5312576, -60.0037346, 76.5386429, -105.6953735, 100.5349884
4: -28.0310745, 39.7003441, -54.9168472, 76.8910980, -104.9221725, 94.6171875

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5206405, upper bound: 96.5571823
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5250360, upper bound: 96.5598081
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5250360, upper bound: 96.5598081
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -42.2711258, 72.1374664, -28.8420296, 49.0686340, -91.3397598, 100.9794846
1: -46.2150154, 60.9471130, -31.5303421, 42.1641083, -88.3791199, 92.4774323
2: -47.3692055, 61.0185242, -32.2884674, 42.0082550, -89.3774567, 93.3069916
3: -54.5314026, 70.6138153, -37.2237091, 48.8474655, -103.3788681, 107.8375244
4: -50.1021652, 70.6908722, -34.5775185, 48.5715103, -98.6736603, 105.2683868

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5952713, upper bound: 96.5975644
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5952713, upper bound: 96.5976979
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -42.2045441, 72.0594559, -29.0617466, 49.4234810, -91.6280212, 101.1211853
1: -46.1434402, 60.8729820, -31.7731133, 42.4179573, -88.5613861, 92.6460876
2: -47.2966080, 60.9421692, -32.5306854, 42.2625122, -89.5591125, 93.4728241
3: -54.4494896, 70.5284958, -37.4984436, 49.1396255, -103.5891113, 108.0269394
4: -50.0332870, 70.5997009, -34.8295670, 48.8867416, -98.9200287, 105.4292679

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5965256, upper bound: 96.5987456
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5965256, upper bound: 96.5987456
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -44.7056580, 74.5382843, -28.4263916, 48.3429298, -93.0485840, 102.9646606
1: -48.8465652, 63.5449524, -31.0648746, 41.5287285, -90.3752823, 94.6098251
2: -50.0575027, 63.6693344, -31.8146992, 41.3796768, -91.4371643, 95.4840317
3: -57.5736580, 73.7311249, -36.6625061, 48.1283417, -105.7019958, 110.3936310
4: -52.8008232, 73.9833755, -34.0804634, 47.8350677, -100.6358948, 108.0638428

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5735859, upper bound: 96.5923860
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5960099, upper bound: 96.5964798
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -44.7056580, 74.5382843, -29.9193306, 49.7362480, -94.4419098, 104.4576111
1: -48.8465652, 63.5449524, -32.6900291, 43.0836143, -91.9301529, 96.2349854
2: -50.0575027, 63.6693344, -33.4670944, 42.9590225, -93.0165253, 97.1364288
3: -57.5736580, 73.7311249, -38.5348320, 49.9806900, -107.5543442, 112.2659607
4: -52.8008232, 73.9833755, -35.7339516, 49.7632103, -102.5640335, 109.7173309

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5735859, upper bound: 96.5937066
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5960099, upper bound: 96.5965135
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -44.4433289, 75.4934387, -21.3757019, 41.6335678, -86.0768967, 96.8691330
1: -48.5721359, 63.7915649, -23.4765701, 34.2805099, -82.8526459, 87.2681351
2: -49.7805099, 63.9304848, -24.1293182, 33.9606133, -83.7411194, 88.0597916
3: -57.2882996, 73.8767395, -28.1568890, 39.6330528, -96.9213333, 102.0336227
4: -52.4813309, 74.0782013, -27.1561260, 38.7777100, -91.2590179, 101.2343292

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5188158, upper bound: 96.5080599
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5563263, upper bound: 96.5170605
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -44.4433289, 75.4934387, -44.5947342, 75.6142807, -120.0576096, 120.0881653
1: -48.5721359, 63.7915649, -48.7442513, 63.9429054, -112.5150452, 112.5358124
2: -49.7805099, 63.9304848, -49.9570389, 64.0909805, -113.8714752, 113.8875122
3: -57.2882996, 73.8767395, -57.4993515, 74.0581131, -131.3464050, 131.3760986
4: -52.4813309, 74.0782013, -52.6233177, 74.2953873, -126.7767105, 126.7015152

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5188161, upper bound: 96.5352946
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5563268, upper bound: 96.5974030
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -46.4532967, 77.2584229, -33.8435516, 56.4931412, -102.9464417, 111.1019745
1: -50.7568550, 65.8649826, -36.9918480, 48.9531441, -99.7099991, 102.8568115
2: -51.9858818, 66.0737762, -37.8853111, 48.8672714, -100.8531494, 103.9590836
3: -59.8171692, 76.3427658, -43.6035500, 56.8591499, -116.6763077, 119.9463120
4: -54.6401863, 76.7295074, -40.2203445, 56.5947151, -111.2349014, 116.9498520

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5576856, upper bound: 96.5773858
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5947393, upper bound: 96.5857644
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -46.5171051, 77.4090118, -32.8311234, 54.7278442, -101.2449341, 110.2401352
1: -50.8310242, 66.0036621, -35.8803940, 47.2792892, -98.1103058, 101.8840561
2: -52.0778999, 66.1841354, -36.7543793, 47.1926575, -99.2705460, 102.9385147
3: -59.9195366, 76.5426712, -42.2875633, 54.8402405, -114.7597809, 118.8302307
4: -54.7891769, 76.8967285, -39.0307922, 54.6572876, -109.4464569, 115.9275131

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5590680, upper bound: 96.5877409
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5966255, upper bound: 96.5966256
time: 0.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.41 seconds
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.4760176, upper bound: 96.5438975
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.4927647, upper bound: 96.5444218
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5238321, upper bound: 96.5590107
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5238321, upper bound: 96.5590107
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5250360, upper bound: 96.5598081
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5250360, upper bound: 96.5598081
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5952713, upper bound: 96.5975644
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5952713, upper bound: 96.5976979
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5965256, upper bound: 96.5987456
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5965256, upper bound: 96.5987456
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5735859, upper bound: 96.5923860
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5960099, upper bound: 96.5964798
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5735859, upper bound: 96.5937066
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5960099, upper bound: 96.5965135
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5188158, upper bound: 96.5080599
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5563263, upper bound: 96.5170605
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5188161, upper bound: 96.5352946
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5563268, upper bound: 96.5974030
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5576856, upper bound: 96.5773858
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5947393, upper bound: 96.5857644
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5590680, upper bound: 96.5877409
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 4, lower bound: -96.5966255, upper bound: 96.5966256

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -18.3819046, 36.9695129, -46.2862701, 78.5400848, -96.9219894, 83.2557831
1: -20.2664185, 30.4266930, -50.5628662, 66.6327744, -86.8991852, 80.9895248
2: -20.8394241, 30.1226788, -51.9070892, 66.6786118, -87.5180359, 82.0297699
3: -24.4666691, 35.2315140, -59.6122169, 77.1233673, -101.5900345, 94.8437347
4: -23.9793987, 34.3001366, -54.8386307, 77.2490463, -101.2284470, 89.1387634

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4805139, upper bound: 96.5228279
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4805138, upper bound: 96.5438975
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -18.4370003, 37.1191292, -46.2021484, 78.4432220, -96.8802185, 83.3212738
1: -20.3259125, 30.4757500, -50.4728012, 66.5390396, -86.8649445, 80.9485474
2: -20.8909130, 30.1708527, -51.8151855, 66.5813904, -87.4723053, 81.9860229
3: -24.5299473, 35.2841492, -59.5103607, 77.0149536, -101.5448914, 94.7945023
4: -24.0247593, 34.3762627, -54.7478256, 77.1326904, -101.1574478, 89.1240845

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4852035, upper bound: 96.5233823
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4852035, upper bound: 96.5444223
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.0142288, 41.1959839, -44.3970337, 75.3124924, -96.3267136, 85.5930176
1: -23.0874786, 33.8037224, -48.4995232, 63.6381607, -86.7256393, 82.3032227
2: -23.7088909, 33.4833374, -49.7060699, 63.7604141, -87.4692764, 83.1894073
3: -27.6857224, 39.0641861, -57.1624527, 73.7039413, -101.3896637, 96.2266388
4: -26.7909698, 38.2138977, -52.4288597, 73.8736267, -100.6645966, 90.6427536

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5077919, upper bound: 96.5185302
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5170294, upper bound: 96.5557727
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -23.0228920, 42.9169579, -44.3970337, 75.3124924, -98.3353882, 87.3139954
1: -25.2076340, 35.7910118, -48.4995232, 63.6381607, -88.8457947, 84.2905350
2: -25.9057255, 35.5229988, -49.7060699, 63.7604141, -89.6661224, 85.2290649
3: -30.0835152, 41.4569817, -57.1624527, 73.7039413, -103.7874527, 98.6194305
4: -28.8122578, 40.6812859, -52.4288597, 73.8736267, -102.6858826, 93.1101456

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5077919, upper bound: 96.5185928
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5170294, upper bound: 96.5557727
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.0142288, 41.1959839, -46.6879501, 77.4099503, -98.4241714, 87.8839264
1: -23.0874786, 33.8037224, -50.9693413, 66.0052872, -89.0927582, 84.7730408
2: -23.7088909, 33.4833374, -52.2223511, 66.1905823, -89.8994751, 85.7056885
3: -27.6857224, 39.0641861, -60.0037346, 76.5386429, -104.2243652, 99.0679092
4: -26.7909698, 38.2138977, -54.9168472, 76.8910980, -103.6820679, 93.1307449

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5077919, upper bound: 96.5437155
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5170294, upper bound: 96.5558645
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.3465767, 43.1221008, -46.6879501, 77.4099503, -100.7565308, 89.8100433
1: -25.5338821, 35.9843025, -50.9693413, 66.0052872, -91.5391541, 86.9536285
2: -26.2375641, 35.7268143, -52.2223511, 66.1905823, -92.4281464, 87.9491653
3: -30.3952370, 41.6864510, -60.0037346, 76.5386429, -106.9338837, 101.6901703
4: -29.0679703, 40.9288330, -54.9168472, 76.8910980, -105.9590683, 95.8456802

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5077919, upper bound: 96.5296947
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5170294, upper bound: 96.5557732
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -41.9705582, 71.8055420, -28.8420296, 49.0686340, -91.0391922, 100.6475449
1: -45.8910713, 60.6282272, -31.5303421, 42.1641083, -88.0551605, 92.1585541
2: -47.0417633, 60.6916733, -32.2884674, 42.0082550, -89.0500183, 92.9801407
3: -54.1728172, 70.2325439, -37.2237091, 48.8474655, -103.0202789, 107.4562531
4: -49.7741013, 70.2923584, -34.5775185, 48.5715103, -98.3455963, 104.8698730

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5946380, upper bound: 96.5961897
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5946380, upper bound: 96.5975644
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -42.1452560, 72.1116333, -28.8420296, 49.0686340, -91.2138901, 100.9536285
1: -46.0837135, 60.8346062, -31.5303421, 42.1641083, -88.2478180, 92.3649445
2: -47.2345505, 60.8984718, -32.2884674, 42.0082550, -89.2428055, 93.1869354
3: -54.3903847, 70.4781265, -37.2237091, 48.8474655, -103.2378464, 107.7018356
4: -49.9825706, 70.5552902, -34.5775185, 48.5715103, -98.5540771, 105.1328125

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5946380, upper bound: 96.5961897
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5946380, upper bound: 96.5976979
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -39.7812042, 67.8594513, -29.0617466, 49.4234810, -89.2046814, 96.9211960
1: -43.4490471, 57.1525650, -31.7731133, 42.4179573, -85.8669891, 88.9256592
2: -44.5035744, 57.2045860, -32.5306854, 42.2625122, -86.7660828, 89.7352448
3: -51.1528740, 66.1774902, -37.4984436, 49.1396255, -100.2924957, 103.6759338
4: -47.1016617, 66.1943817, -34.8295670, 48.8867416, -95.9883881, 101.0239487

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5959870, upper bound: 96.5981837
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5959870, upper bound: 96.5987456
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -43.0248489, 73.5755692, -29.0617466, 49.4234810, -92.4483185, 102.6372986
1: -47.0470848, 62.0623932, -31.7731133, 42.4179573, -89.4650421, 93.8354797
2: -48.2233810, 62.1766701, -32.5306854, 42.2625122, -90.4858932, 94.7073517
3: -55.5564003, 71.8578262, -37.4984436, 49.1396255, -104.6960297, 109.3562698
4: -50.8983841, 72.0122681, -34.8295670, 48.8867416, -99.7851105, 106.8418350

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5959870, upper bound: 96.5981837
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5959870, upper bound: 96.5987456
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -43.9879494, 73.7604446, -27.3280182, 46.7774506, -90.7653961, 101.0884628
1: -48.1024055, 63.0409584, -29.8823280, 40.1558075, -88.2582092, 92.9232788
2: -49.2788887, 63.1180000, -30.6020756, 40.0015182, -89.2804108, 93.7200775
3: -56.7545738, 73.1169128, -35.3133240, 46.5339622, -103.2885056, 108.4302368
4: -52.0391083, 73.2903519, -32.8363190, 46.1907616, -98.2298584, 106.1266708

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5626016, upper bound: 96.5889682
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5562441, upper bound: 96.5717328
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -43.3156548, 72.5494537, -27.8604660, 47.4933243, -90.8089752, 100.4099045
1: -47.3581276, 61.8218536, -30.4579964, 40.7969208, -88.1550369, 92.2798309
2: -48.5273590, 61.9066620, -31.1907959, 40.6489677, -89.1763306, 93.0974579
3: -55.8645935, 71.7450104, -35.9682655, 47.2983360, -103.1629257, 107.7132721
4: -51.2558975, 71.9057846, -33.4445190, 46.9612427, -98.2171402, 105.3503036

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5944257, upper bound: 96.5934248
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5946577, upper bound: 96.5952963
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -43.9879494, 73.7604446, -28.8540192, 48.2049026, -92.1928482, 102.6144638
1: -48.1024055, 63.0409584, -31.5382042, 41.7452126, -89.8476105, 94.5791550
2: -49.2788887, 63.1180000, -32.2895584, 41.6189117, -90.8977966, 95.4075623
3: -56.7545738, 73.1169128, -37.2155609, 48.4327431, -105.1873093, 110.3324738
4: -52.0391083, 73.2903519, -34.5212822, 48.1639214, -100.2030258, 107.8116302

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5632751, upper bound: 96.5890654
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5849443, upper bound: 96.5826736
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5858814, upper bound: 96.5934546
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -43.3156548, 72.5494537, -29.3683338, 48.9113998, -92.2270355, 101.9177856
1: -47.3581276, 61.8218536, -32.0981407, 42.3758469, -89.7339630, 93.9199600
2: -48.5273590, 61.9066620, -32.8595505, 42.2519417, -90.7792969, 94.7661972
3: -55.8645935, 71.7450104, -37.8560600, 49.1815948, -105.0461655, 109.6010513
4: -51.2558975, 71.9057846, -35.1087532, 48.9161758, -100.1720734, 107.0145264

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5947846, upper bound: 96.5936650
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5949315, upper bound: 96.5950496
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -43.4398575, 75.1507874, -20.6968384, 40.7510529, -84.1909103, 95.8476257
1: -47.5383415, 63.2247810, -22.7427025, 33.4477615, -80.9860992, 85.9674683
2: -48.7420731, 63.2887535, -23.3854160, 33.1389008, -81.8809738, 86.6741638
3: -56.2948952, 73.1667557, -27.3382130, 38.6498909, -94.9447708, 100.5049667
4: -51.5811195, 73.2652054, -26.4069805, 37.8084564, -89.3895721, 99.6721878

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5563263, upper bound: 96.5170605
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5563263, upper bound: 96.5170605
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -42.2674789, 72.6221924, -43.6182289, 74.3412933, -116.6087723, 116.2404099
1: -46.2114182, 61.1011200, -47.6871872, 62.7460899, -108.9575043, 108.7883072
2: -47.3888130, 61.2144012, -48.8856659, 62.8824043, -110.2712097, 110.1000671
3: -54.5895042, 70.7270279, -56.2935638, 72.6565323, -127.2460327, 127.0205765
4: -50.0655403, 70.8193970, -51.5480499, 72.8418121, -122.9073334, 122.3674393

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5228609, upper bound: 96.5292351
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5229063, upper bound: 96.5334249
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -43.4398575, 75.1507874, -43.6357994, 74.6049194, -118.0447693, 118.7865906
1: -47.5383415, 63.2247810, -47.7257881, 62.9926682, -110.5309906, 110.9505692
2: -48.7420731, 63.2887535, -48.9305496, 63.1225777, -111.8646545, 112.2192993
3: -56.2948952, 73.1667557, -56.3920441, 72.9454117, -129.2403107, 129.5588074
4: -51.5811195, 73.2652054, -51.6448364, 73.1195526, -124.7006683, 124.9100418

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5895836, upper bound: 96.5712254
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5929976, upper bound: 96.5930981
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -44.3626480, 74.5045929, -32.6077919, 55.0087509, -99.3713913, 107.1123810
1: -48.4903145, 63.2819099, -35.6613083, 47.5386162, -96.0289154, 98.9431915
2: -49.6802292, 63.4850922, -36.5394058, 47.4501114, -97.1303406, 100.0244751
3: -57.2295151, 73.3009491, -42.1168480, 55.1688881, -112.3983765, 115.4177933
4: -52.2926674, 73.6142120, -38.8676682, 54.8863831, -107.1790466, 112.4818802

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5399691, upper bound: 96.5120890
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5399697, upper bound: 96.5656718
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -45.2613106, 76.5504684, -32.9489594, 55.3786011, -100.6399078, 109.4994278
1: -49.5055199, 64.9627991, -36.0240517, 47.9110298, -97.4165344, 100.9868469
2: -50.7151604, 65.1115265, -36.9031219, 47.8201904, -98.5353546, 102.0146408
3: -58.5372238, 75.2356644, -42.5198174, 55.6420441, -114.1792679, 117.7554550
4: -53.4958344, 75.4997864, -39.2240524, 55.3665810, -108.8623886, 114.7238312

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5523818, upper bound: 96.5151033
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5523823, upper bound: 96.5857644
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -44.4186096, 74.6396408, -31.6641922, 53.3236847, -97.7422943, 106.3038330
1: -48.5560112, 63.3921432, -34.6257019, 45.9415703, -94.4975662, 98.0178375
2: -49.7571907, 63.5814133, -35.4820099, 45.8550415, -95.6122284, 99.0634232
3: -57.3217430, 73.4610519, -40.8869781, 53.2514763, -110.5732040, 114.3480148
4: -52.4075546, 73.7639236, -37.7566719, 53.0580139, -105.4655533, 111.5205994

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5405383, upper bound: 96.5144163
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5405389, upper bound: 96.5712250
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -45.4995766, 76.8106766, -31.8673058, 53.5467796, -99.0463486, 108.6779785
1: -49.7559204, 65.1840057, -34.8365402, 46.1708031, -95.9266968, 100.0205460
2: -50.9726715, 65.3331604, -35.7020493, 46.0777817, -97.0504532, 101.0352097
3: -58.8128891, 75.5200272, -41.1209412, 53.5455627, -112.3584518, 116.6409683
4: -53.7586594, 75.8003998, -37.9661331, 53.3305817, -107.0892258, 113.7665329

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5539154, upper bound: 96.5179926
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5539159, upper bound: 96.5966256
time: 0.88 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.94 seconds
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.4805139, upper bound: 96.5228279
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.4805138, upper bound: 96.5438975
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.4852035, upper bound: 96.5233823
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.4852035, upper bound: 96.5444223
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5077919, upper bound: 96.5185302
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5170294, upper bound: 96.5557727
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5077919, upper bound: 96.5185928
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5170294, upper bound: 96.5557727
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5077919, upper bound: 96.5437155
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5170294, upper bound: 96.5558645
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5077919, upper bound: 96.5296947
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5170294, upper bound: 96.5557732
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5946380, upper bound: 96.5961897
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5946380, upper bound: 96.5975644
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5946380, upper bound: 96.5961897
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5946380, upper bound: 96.5976979
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5959870, upper bound: 96.5981837
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5959870, upper bound: 96.5987456
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5959870, upper bound: 96.5981837
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5959870, upper bound: 96.5987456
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5626016, upper bound: 96.5889682
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5562441, upper bound: 96.5717328
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5944257, upper bound: 96.5934248
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5946577, upper bound: 96.5952963
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5849443, upper bound: 96.5826736
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5858814, upper bound: 96.5934546
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5947846, upper bound: 96.5936650
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5949315, upper bound: 96.5950496
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5563263, upper bound: 96.5170605
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5563263, upper bound: 96.5170605
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5228609, upper bound: 96.5292351
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5229063, upper bound: 96.5334249
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5895836, upper bound: 96.5712254
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5929976, upper bound: 96.5930981
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5399691, upper bound: 96.5120890
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5399697, upper bound: 96.5656718
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5523818, upper bound: 96.5151033
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5523823, upper bound: 96.5857644
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5405383, upper bound: 96.5144163
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5405389, upper bound: 96.5712250
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5539154, upper bound: 96.5179926
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.94
Output dim: 4, lower bound: -96.5539159, upper bound: 96.5966256

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -18.3819046, 36.9695129, -43.7424660, 74.3729630, -92.7548676, 80.7119751
1: -20.2664185, 30.4266930, -47.8143196, 63.3269730, -83.5933914, 78.2410126
2: -20.8394241, 30.1226788, -49.0947533, 63.3423729, -84.1817780, 79.2174301
3: -24.4666691, 35.2315140, -56.4541779, 73.2730560, -97.7397232, 91.6856842
4: -23.9793987, 34.3001366, -52.0857544, 73.3504257, -97.3298264, 86.3858948

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4748822, upper bound: 96.5189829
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4777947, upper bound: 96.5226628
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -18.3819046, 36.9695129, -48.3444557, 80.9938583, -99.3757629, 85.3139648
1: -20.2664185, 30.4266930, -52.7870903, 68.9757843, -89.2421951, 83.2137833
2: -20.8394241, 30.1226788, -54.1453247, 69.0867920, -89.9262161, 84.2680054
3: -24.4666691, 35.2315140, -62.1847153, 79.8348694, -104.3015366, 97.4162140
4: -23.9793987, 34.3001366, -57.0075569, 80.1220703, -104.1014709, 91.3076935

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4748822, upper bound: 96.5402352
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4777947, upper bound: 96.5226628
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -18.4370003, 37.1191292, -43.6517105, 74.2661285, -92.7031250, 80.7708282
1: -20.3259125, 30.4757500, -47.7170334, 63.2224731, -83.5483780, 78.1927795
2: -20.8909130, 30.1708527, -48.9950294, 63.2365227, -84.1274338, 79.1658783
3: -24.5299473, 35.2841492, -56.3441315, 73.1521149, -97.6820602, 91.6282578
4: -24.0247593, 34.3762627, -51.9852715, 73.2238007, -97.2485580, 86.3615341

Time for backsubstitution: 2.21 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=108.20734405517578
rel_dist={4: [-96.6182045307215, 96.61820453072153]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1115.15 seconds
