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
execution time: IAR + LP analysis = 2.17 + 2.21 = 4.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -96.6185813, upper bound: 96.6185813


# Binary Search by BASE starts (time budget: 1195.62 seconds, max iter: 100)

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
Binary search time: 81.24 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1114.38 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6017281, upper bound: 96.5315669
time: 0.91 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6017281, upper bound: 96.5315669
time: 1.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.25 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 2.25
Output dim: 4, lower bound: -96.6017281, upper bound: 96.5315669
IS_B2, status: Status.UNKNOWN, split count: 1, time: 2.25
Output dim: 4, lower bound: -96.6017281, upper bound: 96.5315669

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -38.2100830, 62.7565804, -23.6491947, 44.5811996, -82.7912750, 86.4057693
1: -41.6995087, 54.3409386, -25.9020424, 36.9293671, -78.6288757, 80.2429733
2: -42.7050247, 54.2436867, -26.6062851, 36.6187668, -79.3237610, 80.8499603
3: -48.9462700, 63.1773834, -30.8861237, 42.7392807, -91.6855469, 94.0635071
4: -45.2550850, 62.9522591, -29.5871983, 41.9198265, -87.1749115, 92.5394592

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5949495, upper bound: 96.5306942
time: 0.90 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5949495, upper bound: 96.5313641
time: 0.85 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -38.0210800, 62.4870491, -47.7588348, 79.9911652, -118.0122452, 110.2458725
1: -41.4938278, 54.0912552, -52.1639366, 67.9121017, -109.4059219, 106.2551804
2: -42.4946556, 53.9964714, -53.4510193, 68.0761642, -110.5708160, 107.4474945
3: -48.7111588, 62.8709106, -61.4133224, 78.7076035, -127.4187622, 124.2842255
4: -45.0321922, 62.6598892, -56.2418251, 79.0030441, -124.0352325, 118.9017029

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5315669, upper bound: 96.6017281
time: 0.74 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5315669, upper bound: 96.6152409
time: 0.92 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.73 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 4, lower bound: -96.5949495, upper bound: 96.5306942
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 4, lower bound: -96.5949495, upper bound: 96.5313641
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 4, lower bound: -96.5315669, upper bound: 96.6017281
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 4, lower bound: -96.5315669, upper bound: 96.6152409

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -37.5779495, 62.1574745, -23.6431980, 44.5752335, -82.1531601, 85.8006744
1: -41.0252304, 53.7039070, -25.8958530, 36.9236984, -77.9489288, 79.5997620
2: -42.0214386, 53.5866394, -26.5999107, 36.6129456, -78.6343842, 80.1865463
3: -48.2054901, 62.4242668, -30.8795776, 42.7325821, -90.9380722, 93.3038254
4: -44.5850372, 62.1696434, -29.5810375, 41.9129829, -86.4980164, 91.7506790

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4764256, upper bound: 96.5117448
time: 0.86 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4764256, upper bound: 96.5117448
time: 0.88 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -38.1697311, 62.7158737, -23.6491947, 44.5811996, -82.7509308, 86.3650665
1: -41.6565094, 54.3016815, -25.9020424, 36.9293671, -78.5858765, 80.2037201
2: -42.6617813, 54.2028618, -26.6062851, 36.6187668, -79.2805481, 80.8091431
3: -48.8994026, 63.1311684, -30.8861237, 42.7392807, -91.6386871, 94.0172882
4: -45.2133522, 62.9031982, -29.5871983, 41.9198265, -87.1331711, 92.4903946

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5178301, upper bound: 96.5178312
time: 0.98 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5178301, upper bound: 96.5313641
time: 0.91 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -23.6491947, 44.5811996, -47.7588348, 79.9911652, -103.6403580, 92.3400269
1: -25.9020424, 36.9293671, -52.1639366, 67.9121017, -93.8141251, 89.0933075
2: -26.6062851, 36.6187668, -53.4510193, 68.0761642, -94.6824493, 90.0697861
3: -30.8861237, 42.7392807, -61.4133224, 78.7076035, -109.5937271, 104.1525955
4: -29.5871983, 41.9198265, -56.2418251, 79.0030441, -108.5902405, 98.1616516

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5117448, upper bound: 96.5949495
time: 0.90 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5178312, upper bound: 96.6014877
time: 0.72 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -47.7588348, 79.9911652, -47.7588348, 79.9911652, -127.7499771, 127.7499771
1: -52.1639366, 67.9121017, -52.1639366, 67.9121017, -120.0760345, 120.0760345
2: -53.4510193, 68.0761642, -53.4510193, 68.0761642, -121.5271835, 121.5271835
3: -61.4133224, 78.7076035, -61.4133224, 78.7076035, -140.1209106, 140.1208954
4: -56.2418251, 79.0030441, -56.2418251, 79.0030441, -135.2448730, 135.2448730

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5114557, upper bound: 96.5141563
time: 0.74 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5180471, upper bound: 96.5180471
time: 0.93 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.79 seconds
IS_B1_A1_A1, status: Status.VERIFIED, split count: 3, time: 3.79
Output dim: 4, lower bound: -96.4764256, upper bound: 96.5117448
IS_B1_A1_A2, status: Status.VERIFIED, split count: 3, time: 3.79
Output dim: 4, lower bound: -96.4764256, upper bound: 96.5117448
IS_B1_A2_A1, status: Status.VERIFIED, split count: 3, time: 3.79
Output dim: 4, lower bound: -96.5178301, upper bound: 96.5178312
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 4, lower bound: -96.5178301, upper bound: 96.5313641
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 4, lower bound: -96.5117448, upper bound: 96.5949495
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.79
Output dim: 4, lower bound: -96.5178312, upper bound: 96.6014877
IS_B2_A2_A1, status: Status.VERIFIED, split count: 3, time: 3.79
Output dim: 4, lower bound: -96.5114557, upper bound: 96.5141563
IS_B2_A2_A2, status: Status.VERIFIED, split count: 3, time: 3.79
Output dim: 4, lower bound: -96.5180471, upper bound: 96.5180471

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -47.7190552, 79.9503479, -23.6491947, 44.5811996, -92.3002472, 103.5995331
1: -52.1214371, 67.8727264, -25.9020424, 36.9293671, -89.0508041, 93.7747421
2: -53.4082642, 68.0354233, -26.6062851, 36.6187668, -90.0270309, 94.6417084
3: -61.3669090, 78.6607513, -30.8861237, 42.7392807, -104.1061859, 109.5468750
4: -56.2001648, 78.9533081, -29.5871983, 41.9198265, -98.1199799, 108.5405045

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5112749, upper bound: 96.5139443
time: 0.93 seconds

## Relational analysis of IS_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5178242, upper bound: 96.5310967
time: 0.86 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -23.6431980, 44.5752335, -47.1010628, 79.3604965, -103.0036926, 91.6762924
1: -25.8958530, 36.9236984, -51.4631310, 67.2346344, -93.1304855, 88.3868256
2: -26.5999107, 36.6129456, -52.7382889, 67.3822784, -93.9821777, 89.3512344
3: -30.8795776, 42.7325821, -60.6423721, 77.9089737, -108.7885513, 103.3749542
4: -29.5810375, 41.9129829, -55.5394173, 78.1715088, -107.7525482, 97.4524002

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5292647, upper bound: 96.5828428
time: 0.94 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5303360, upper bound: 96.5862680
time: 0.82 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -23.6491947, 44.5811996, -47.7190552, 79.9503479, -103.5995331, 92.3002548
1: -25.9020424, 36.9293671, -52.1214371, 67.8727264, -93.7747498, 89.0507965
2: -26.6062851, 36.6187668, -53.4082642, 68.0354233, -94.6417084, 90.0270309
3: -30.8861237, 42.7392807, -61.3669090, 78.6607513, -109.5468750, 104.1061859
4: -29.5871983, 41.9198265, -56.2001648, 78.9533081, -108.5405045, 98.1199799

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5300904, upper bound: 96.5912010
time: 0.78 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5310967, upper bound: 96.5924613
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.70 seconds
IS_B1_A2_A2_A1, status: Status.VERIFIED, split count: 4, time: 3.70
Output dim: 4, lower bound: -96.5112749, upper bound: 96.5139443
IS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 4, lower bound: -96.5178242, upper bound: 96.5310967
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 4, lower bound: -96.5292647, upper bound: 96.5828428
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 4, lower bound: -96.5303360, upper bound: 96.5862680
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 4, lower bound: -96.5300904, upper bound: 96.5912010
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 4, lower bound: -96.5310967, upper bound: 96.5924613

## BFS IS instance: IS_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -45.7679825, 77.4062424, -23.6491947, 44.5811996, -90.3491821, 101.0554352
1: -50.0280113, 65.5433350, -25.9020424, 36.9293671, -86.9573822, 91.4453583
2: -51.2725220, 65.7026978, -26.6062851, 36.6187668, -87.8912659, 92.3089752
3: -59.0105782, 75.9060974, -30.8861237, 42.7392807, -101.7498627, 106.7922211
4: -53.9547539, 76.1816177, -29.5871983, 41.9198265, -95.8745804, 105.7688141

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5804082, upper bound: 96.5007197
time: 1.07 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5922848, upper bound: 96.5310667
time: 0.95 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -23.3921528, 44.2467728, -40.8853493, 69.7066422, -93.0987930, 85.1321259
1: -25.6274567, 36.6235962, -44.6765060, 58.7597847, -84.3872299, 81.3001022
2: -26.3239269, 36.3109818, -45.7650757, 58.7995110, -85.1234360, 82.0760574
3: -30.5752487, 42.3804970, -52.6343346, 68.0640945, -98.6393433, 95.0148087
4: -29.3062000, 41.5591507, -48.4212799, 68.0924225, -97.3986206, 89.9804306

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4987509, upper bound: 96.5707897
time: 0.98 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5290489, upper bound: 96.5826663
time: 1.20 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -23.6431980, 44.5752335, -45.1577072, 76.8224716, -100.4656677, 89.7329407
1: -25.8958530, 36.9236984, -49.3774605, 64.9149094, -90.8107605, 86.3011322
2: -26.5999107, 36.6129456, -50.6108704, 65.0580444, -91.6579590, 87.2238159
3: -30.8795776, 42.7325821, -58.2938576, 75.1658859, -106.0454636, 101.0264435
4: -29.5810375, 41.9129829, -53.3054352, 75.4073257, -104.9883652, 95.2184143

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4999375, upper bound: 96.5742150
time: 0.77 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5303126, upper bound: 96.5860916
time: 0.77 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -23.3981304, 44.2527313, -41.4901619, 70.2804794, -93.6786118, 85.7428894
1: -25.6336288, 36.6292534, -45.3206100, 59.3773613, -85.0109863, 81.9498444
2: -26.3302841, 36.3167763, -46.4202805, 59.4345970, -85.7648773, 82.7370300
3: -30.5817738, 42.3871727, -53.3425140, 68.7938156, -99.3755722, 95.7296906
4: -29.3123360, 41.5659599, -49.0683098, 68.8546600, -98.1669922, 90.6342697

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4995006, upper bound: 96.5791479
time: 1.05 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5297985, upper bound: 96.5910245
time: 0.86 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -23.6491947, 44.5811996, -45.7679825, 77.4062424, -101.0554352, 90.3491821
1: -25.9020424, 36.9293671, -50.0280113, 65.5433350, -91.4453583, 86.9573746
2: -26.6062851, 36.6187668, -51.2725220, 65.7026978, -92.3089752, 87.8912582
3: -30.8861237, 42.7392807, -59.0105782, 75.9060974, -106.7922211, 101.7498627
4: -29.5871983, 41.9198265, -53.9547539, 76.1816177, -105.7688141, 95.8745728

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5007197, upper bound: 96.5804082
time: 0.93 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5310667, upper bound: 96.5922848
time: 0.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.86 seconds
IS_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 4, lower bound: -96.5804082, upper bound: 96.5007197
IS_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 4, lower bound: -96.5922848, upper bound: 96.5310667
IS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 4, lower bound: -96.4987509, upper bound: 96.5707897
IS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 4, lower bound: -96.5290489, upper bound: 96.5826663
IS_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 4, lower bound: -96.4999375, upper bound: 96.5742150
IS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 4, lower bound: -96.5303126, upper bound: 96.5860916
IS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 4, lower bound: -96.4995006, upper bound: 96.5791479
IS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 4, lower bound: -96.5297985, upper bound: 96.5910245
IS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 4, lower bound: -96.5007197, upper bound: 96.5804082
IS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 4, lower bound: -96.5310667, upper bound: 96.5922848

## BFS IS instance: IS_B1_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -45.7679825, 77.4062424, -19.0168400, 37.8047943, -83.5727692, 96.4230804
1: -50.0280113, 65.5433350, -20.9385338, 31.1538448, -81.1818542, 86.4818573
2: -51.2725220, 65.7026978, -21.5286846, 30.8517284, -82.1242447, 87.2313843
3: -59.0105782, 75.9060974, -25.2066669, 36.0842285, -95.0948029, 101.1127625
4: -53.9547539, 76.1816177, -24.6514683, 35.1617165, -89.1164627, 100.8330841

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5785689, upper bound: 96.4956574
time: 0.88 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B2

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5788595, upper bound: 96.4976957
time: 0.93 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -45.7679825, 77.4062424, -22.9655075, 43.7550316, -89.5230103, 100.3717499
1: -50.0280113, 65.5433350, -25.1754646, 36.1651459, -86.1931534, 90.7187881
2: -51.2725220, 65.7026978, -25.8641987, 35.8516693, -87.1241760, 91.5668793
3: -59.0105782, 75.9060974, -30.0730324, 41.8423233, -100.8529053, 105.9791183
4: -53.9547539, 76.1816177, -28.8824005, 41.0101280, -94.9648666, 105.0640182

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_A2_B2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5848485, upper bound: 96.5184613
time: 1.38 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5848486, upper bound: 96.5296189
time: 0.71 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -18.7651711, 37.4635353, -40.8853493, 69.7066422, -88.4718018, 78.3488770
1: -20.6697197, 30.8443527, -44.6765060, 58.7597847, -79.4295044, 75.5208588
2: -21.2511005, 30.5412483, -45.7650757, 58.7995110, -80.0506134, 76.3063202
3: -24.9010048, 35.7227211, -52.6343346, 68.0640945, -92.9651031, 88.3570480
4: -24.3758583, 34.7997475, -48.4212799, 68.0924225, -92.4682770, 83.2210236

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4987326, upper bound: 96.5686972
time: 0.82 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4987509, upper bound: 96.5707897
time: 1.10 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4987509, upper bound: 96.5707897
time: 0.84 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -22.7086411, 43.4216232, -40.8853493, 69.7066422, -92.4152832, 84.3069687
1: -24.9010010, 35.8605080, -44.6765060, 58.7597847, -83.6607819, 80.5370178
2: -25.5821075, 35.5451469, -45.7650757, 58.7995110, -84.3816071, 81.3102188
3: -29.7623997, 41.4847946, -52.6343346, 68.0640945, -97.8264847, 94.1190948
4: -28.6016197, 40.6510620, -48.4212799, 68.0924225, -96.6940384, 89.0723267

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5752300
time: 1.23 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5826663
time: 1.12 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -19.0113773, 37.7993164, -45.1577072, 76.8224716, -95.8338394, 82.9570007
1: -20.9329071, 31.1485634, -49.3774605, 64.9149094, -85.8478165, 80.5260162
2: -21.5228539, 30.8463860, -50.6108704, 65.0580444, -86.5808945, 81.4572449
3: -25.2007027, 36.0779991, -58.2938576, 75.1658859, -100.3665924, 94.3718567
4: -24.6458397, 35.1554260, -53.3054352, 75.4073257, -100.0531616, 88.4608612

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4979210, upper bound: 96.5572611
time: 1.27 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4990640, upper bound: 96.5665094
time: 0.88 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -22.9594498, 43.7490120, -45.1577072, 76.8224716, -99.7819214, 88.9067078
1: -25.1692162, 36.1594276, -49.3774605, 64.9149094, -90.0841217, 85.5368652
2: -25.8577595, 35.8458138, -50.6108704, 65.0580444, -90.9158020, 86.4566803
3: -30.0664291, 41.8355789, -58.2938576, 75.1658859, -105.2323151, 100.1294403
4: -28.8762016, 41.0032349, -53.3054352, 75.4073257, -104.2835236, 94.3086700

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5786554
time: 0.72 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5860916
time: 1.07 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -18.7706032, 37.4689789, -41.4901619, 70.2804794, -89.0510864, 78.9591370
1: -20.6753273, 30.8495636, -45.3206100, 59.3773613, -80.0526886, 76.1701660
2: -21.2569008, 30.5465603, -46.4202805, 59.4345970, -80.6914978, 76.9668274
3: -24.9069366, 35.7289238, -53.3425140, 68.7938156, -93.7007370, 89.0714264
4: -24.3814468, 34.8059959, -49.0683098, 68.8546600, -93.2361069, 83.8743057

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5772027
time: 0.92 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5774933
time: 0.78 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -22.7146873, 43.4276276, -41.4901619, 70.2804794, -92.9951630, 84.9177856
1: -24.9072475, 35.8662186, -45.3206100, 59.3773613, -84.2846069, 81.1867981
2: -25.5885277, 35.5509949, -46.4202805, 59.4345970, -85.0231247, 81.9712524
3: -29.7689934, 41.4915161, -53.3425140, 68.7938156, -98.5628052, 94.8340149
4: -28.6077976, 40.6579285, -49.0683098, 68.8546600, -97.4624557, 89.7262421

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B2_B1_A2_B1

### Relational analysis result of IS_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5169582, upper bound: 96.5528129
time: 0.95 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2_B2

### Relational analysis result of IS_B2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5169582, upper bound: 96.5910245
time: 0.83 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -19.0168400, 37.8047943, -45.7679825, 77.4062424, -96.4230804, 83.5727615
1: -20.9385338, 31.1538448, -50.0280113, 65.5433350, -86.4818649, 81.1818542
2: -21.5286846, 30.8517284, -51.2725220, 65.7026978, -87.2313843, 82.1242371
3: -25.2066669, 36.0842285, -59.0105782, 75.9060974, -101.1127625, 95.0948029
4: -24.6514683, 35.1617165, -53.9547539, 76.1816177, -100.8330841, 89.1164627

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4956574, upper bound: 96.5785689
time: 0.98 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_A2

### Relational analysis result of IS_B2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4976957, upper bound: 96.5788595
time: 1.14 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -22.9655075, 43.7550316, -45.7679825, 77.4062424, -100.3717499, 89.5230103
1: -25.1754646, 36.1651459, -50.0280113, 65.5433350, -90.7187958, 86.1931534
2: -25.8641987, 35.8516693, -51.2725220, 65.7026978, -91.5668793, 87.1241837
3: -30.0730324, 41.8423233, -59.0105782, 75.9060974, -105.9791260, 100.8529053
4: -28.8824005, 41.0101280, -53.9547539, 76.1816177, -105.0640182, 94.9648666

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5171931, upper bound: 96.5848486
time: 0.86 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5171931, upper bound: 96.5835882
time: 0.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.78 seconds
IS_B1_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.5785689, upper bound: 96.4956574
IS_B1_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.5788595, upper bound: 96.4976957
IS_B1_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.5848485, upper bound: 96.5184613
IS_B1_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.5848486, upper bound: 96.5296189
IS_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.4987509, upper bound: 96.5707897
IS_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.4987509, upper bound: 96.5707897
IS_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5752300
IS_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5826663
IS_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.4979210, upper bound: 96.5572611
IS_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.4990640, upper bound: 96.5665094
IS_B2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5786554
IS_B2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5860916
IS_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5772027
IS_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5774933
IS_B2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.5169582, upper bound: 96.5528129
IS_B2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.5169582, upper bound: 96.5910245
IS_B2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.4956574, upper bound: 96.5785689
IS_B2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.4976957, upper bound: 96.5788595
IS_B2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.5171931, upper bound: 96.5848486
IS_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.78
Output dim: 4, lower bound: -96.5171931, upper bound: 96.5835882

## BFS IS instance: IS_B1_A2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -45.7679825, 77.4062424, -18.8045158, 37.6210060, -83.3889923, 96.2107544
1: -50.0280113, 65.5433350, -20.7213821, 30.9791679, -81.0071793, 86.2647171
2: -51.2725220, 65.7026978, -21.3083229, 30.6723366, -81.9448395, 87.0110168
3: -59.0105782, 75.9060974, -24.9816628, 35.8777466, -94.8883209, 100.8877563
4: -53.9547539, 76.1816177, -24.4570980, 34.9476204, -88.9023666, 100.6387177

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5785689, upper bound: 96.4956574
time: 1.09 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B2

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5785689, upper bound: 96.4956574
time: 0.92 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -45.7656784, 77.4034882, -18.8575993, 37.7701225, -83.5357971, 96.2610855
1: -50.0255280, 65.5407181, -20.7793446, 31.0269260, -81.0524445, 86.3200455
2: -51.2699890, 65.7000122, -21.3576927, 30.7190914, -81.9890823, 87.0576935
3: -59.0077667, 75.9030762, -25.0427723, 35.9291649, -94.9369278, 100.9458389
4: -53.9522552, 76.1784439, -24.5008812, 35.0223236, -88.9745560, 100.6793213

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5788595, upper bound: 96.4976957
time: 0.91 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5788595, upper bound: 96.4976957
time: 0.89 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -45.7679825, 77.4062424, -16.5918007, 34.7164307, -80.4844131, 93.9980469
1: -50.0280113, 65.5433350, -18.3224945, 28.1499214, -78.1779099, 83.8658066
2: -51.2725220, 65.7026978, -18.7897148, 27.8247471, -79.0972595, 84.4924011
3: -59.0105782, 75.9060974, -22.1792717, 32.4609222, -91.4714966, 98.0853729
4: -53.9547539, 76.1816177, -21.7783241, 31.6278801, -85.5826340, 97.9599380

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5829966, upper bound: 96.5135999
time: 0.95 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5833005, upper bound: 96.5155373
time: 0.98 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -45.7679825, 77.4062424, -21.5673180, 41.9855995, -87.7535782, 98.9735565
1: -50.0280113, 65.5433350, -23.6898689, 34.6065483, -84.6345367, 89.2331924
2: -51.2725220, 65.7026978, -24.3597431, 34.2851677, -85.5576782, 90.0624390
3: -59.0105782, 75.9060974, -28.4248352, 40.0198708, -99.0304489, 104.3309326
4: -53.9547539, 76.1816177, -27.4042778, 39.1560936, -93.1108475, 103.5858917

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5829966, upper bound: 96.5220685
time: 0.86 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5833005, upper bound: 96.5221098
time: 1.04 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -12.8577709, 29.1089993, -40.8853493, 69.7066422, -82.5644073, 69.9943466
1: -14.3430653, 23.4504185, -44.6765060, 58.7597847, -73.1028442, 68.1269226
2: -14.7033768, 23.1299858, -45.7650757, 58.7995110, -73.5028839, 68.8950500
3: -17.6132488, 27.0557327, -52.6343346, 68.0640945, -85.6773453, 79.6900406
4: -17.8484039, 26.1649666, -48.4212799, 68.0924225, -85.9408264, 74.5862274

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4987326, upper bound: 96.5686972
time: 0.80 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4987509, upper bound: 96.5707142
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.3884991, upper bound: 96.5293527
time: 0.82 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4822124, upper bound: 96.5674864
time: 1.04 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -18.0448837, 36.4902649, -40.8853493, 69.7066422, -87.7515259, 77.3756027
1: -19.9012680, 29.9667130, -44.6765060, 58.7597847, -78.6610336, 74.6432190
2: -20.4647598, 29.6777153, -45.7650757, 58.7995110, -79.2642670, 75.4427948
3: -24.0334492, 34.6880455, -52.6343346, 68.0640945, -92.0975418, 87.3223648
4: -23.5479259, 33.7806816, -48.4212799, 68.0924225, -91.6403503, 82.2019501

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4987326, upper bound: 96.5686972
time: 0.91 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.3884991, upper bound: 96.5519902
time: 1.35 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4822124, upper bound: 96.5674864
time: 0.73 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -16.5866318, 34.7111588, -40.8853493, 69.7066422, -86.2932739, 75.5964813
1: -18.3171921, 28.1450157, -44.6765060, 58.7597847, -77.0769577, 72.8215179
2: -18.7842369, 27.8197651, -45.7650757, 58.7995110, -77.5837479, 73.5848389
3: -22.1736679, 32.4551468, -52.6343346, 68.0640945, -90.2377625, 85.0894470
4: -21.7731361, 31.6220474, -48.4212799, 68.0924225, -89.8655548, 80.0433121

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164251, upper bound: 96.5731375
time: 0.99 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5751599
time: 0.94 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5058461, upper bound: 96.5736130
time: 0.93 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -21.5616837, 41.9797440, -40.8853493, 69.7066422, -91.2683258, 82.8650970
1: -23.6840420, 34.6010284, -44.6765060, 58.7597847, -82.4438095, 79.2775345
2: -24.3537216, 34.2795410, -45.7650757, 58.7995110, -83.1532288, 80.0446167
3: -28.4186306, 40.0133286, -52.6343346, 68.0640945, -96.4827271, 92.6476364
4: -27.3984489, 39.1494598, -48.4212799, 68.0924225, -95.4908600, 87.5707321

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164251, upper bound: 96.5800296
time: 0.86 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5819965
time: 1.11 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5058461, upper bound: 96.5736130
time: 0.98 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -18.9855080, 37.7659760, -44.5353241, 76.0703125, -95.0558167, 82.3013000
1: -20.9055061, 31.1180420, -48.7289047, 64.3829498, -85.2884445, 79.8469391
2: -21.4943161, 30.8161736, -49.9155731, 64.5241928, -86.0184937, 80.7317429
3: -25.1698666, 36.0417824, -57.5462685, 74.5447083, -99.7145767, 93.5880432
4: -24.6163826, 35.1194916, -52.6023560, 74.7267838, -99.3431702, 87.7218246

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4979210, upper bound: 96.5572611
time: 0.82 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4404351, upper bound: 96.5482518
time: 0.94 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4810937, upper bound: 96.5535912
time: 0.97 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -19.0113773, 37.7993164, -43.4514542, 74.5323486, -93.5437241, 81.2507477
1: -20.9329071, 31.1485634, -47.5596313, 62.8722229, -83.8051300, 78.7081909
2: -21.5228539, 30.8463860, -48.7404366, 62.9815750, -84.5044174, 79.5868149
3: -25.2007027, 36.0779991, -56.2256660, 72.8205719, -98.0212708, 92.3036423
4: -24.6458397, 35.1554260, -51.4321785, 72.9620056, -97.6078415, 86.5876007

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4980814, upper bound: 96.5665094
time: 0.80 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4980814, upper bound: 96.5661032
time: 0.94 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -16.5866318, 34.7111588, -45.1577072, 76.8224716, -93.4091034, 79.8688354
1: -18.3171921, 28.1450157, -49.3774605, 64.9149094, -83.2321014, 77.5224686
2: -18.7842369, 27.8197651, -50.6108704, 65.0580444, -83.8422699, 78.4306335
3: -22.1736679, 32.4551468, -58.2938576, 75.1658859, -97.3395538, 90.7490082
4: -21.7731361, 31.6220474, -53.3054352, 75.4073257, -97.1804657, 84.9274826

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164251, upper bound: 96.5776464
time: 1.13 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5785852
time: 1.02 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_A1_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5058461, upper bound: 96.5770383
time: 0.78 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -21.5616837, 41.9797440, -45.1577072, 76.8224716, -98.3841553, 87.1374512
1: -23.6840420, 34.6010284, -49.3774605, 64.9149094, -88.5989456, 83.9784698
2: -24.3537216, 34.2795410, -50.6108704, 65.0580444, -89.4117661, 84.8904037
3: -28.4186306, 40.0133286, -58.2938576, 75.1658859, -103.5845184, 98.3071899
4: -27.3984489, 39.1494598, -53.3054352, 75.4073257, -102.8057709, 92.4548950

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164251, upper bound: 96.5845386
time: 1.48 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5734259
time: 0.85 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_A2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5058461, upper bound: 96.5738318
time: 1.34 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -18.5590210, 37.2858200, -41.4901619, 70.2804794, -88.8395004, 78.7759857
1: -20.4592743, 30.6764393, -45.3206100, 59.3773613, -79.8366165, 75.9970245
2: -21.0373611, 30.3678913, -46.4202805, 59.4345970, -80.4719543, 76.7881622
3: -24.6827736, 35.5234871, -53.3425140, 68.7938156, -93.4765778, 88.8659973
4: -24.1878071, 34.5928688, -49.0683098, 68.8546600, -93.0424652, 83.6611786

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B2_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4948020, upper bound: 96.5754821
time: 1.04 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5772027
time: 0.75 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5772027
time: 0.93 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -18.6198730, 37.4432831, -41.4880981, 70.2780914, -88.8979645, 78.9313812
1: -20.5245628, 30.7323761, -45.3183823, 59.3750916, -79.8996582, 76.0507584
2: -21.0947952, 30.4228191, -46.4180298, 59.4322739, -80.5270691, 76.8408508
3: -24.7522507, 35.5842209, -53.3399887, 68.7911758, -93.5434265, 88.9242096
4: -24.2389984, 34.6768188, -49.0661087, 68.8518524, -93.0908508, 83.7429276

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4968402, upper bound: 96.5757727
time: 1.04 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5774933
time: 1.16 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A2_A2

### Relational analysis result of IS_B2_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5774933
time: 0.80 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -22.7146873, 43.4276276, -35.9497871, 62.7456589, -85.4603424, 79.3774109
1: -24.9072475, 35.8662186, -39.3450012, 52.7411232, -77.6483688, 75.2112122
2: -25.5885277, 35.5509949, -40.3378067, 52.7493210, -78.3378448, 75.8887863
3: -29.7689934, 41.4915161, -46.5233955, 61.0511665, -90.8201599, 88.0149078
4: -28.6077976, 40.6579285, -43.0060921, 60.9456406, -89.5534363, 83.6640091

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4584912, upper bound: 96.5421771
time: 0.77 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5041230, upper bound: 96.5504685
time: 1.20 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -22.7146873, 43.4276276, -40.8318634, 69.4166336, -92.1313095, 84.2594833
1: -24.9072475, 35.8662186, -44.6077271, 58.5923157, -83.4995651, 80.4739380
2: -25.5885277, 35.5509949, -45.6984100, 58.6423302, -84.2308578, 81.2493744
3: -29.7689934, 41.4915161, -52.5269661, 67.8743896, -97.6433868, 94.0184784
4: -28.6077976, 40.6579285, -48.3389206, 67.9102631, -96.5180588, 88.9968491

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5043528, upper bound: 96.5555867
time: 1.02 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5043528, upper bound: 96.5528129
time: 0.76 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -18.8045158, 37.6210060, -45.7679825, 77.4062424, -96.2107544, 83.3889923
1: -20.7213821, 30.9791679, -50.0280113, 65.5433350, -86.2647095, 81.0071793
2: -21.3083229, 30.6723366, -51.2725220, 65.7026978, -87.0110168, 81.9448471
3: -24.9816628, 35.8777466, -59.0105782, 75.9060974, -100.8877563, 94.8883209
4: -24.4570980, 34.9476204, -53.9547539, 76.1816177, -100.6387177, 88.9023666

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5785689
time: 1.02 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_A1_A2

### Relational analysis result of IS_B2_A1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5782260
time: 0.87 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -18.8575993, 37.7701225, -45.7656784, 77.4034882, -96.2610855, 83.5357971
1: -20.7793446, 31.0269260, -50.0255280, 65.5407181, -86.3200607, 81.0524445
2: -21.3576927, 30.7190914, -51.2699890, 65.7000122, -87.0577087, 81.9890823
3: -25.0427723, 35.9291649, -59.0077667, 75.9030762, -100.9458389, 94.9369278
4: -24.5008812, 35.0223236, -53.9522552, 76.1784439, -100.6793213, 88.9745483

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5788595
time: 0.98 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5782398
time: 0.76 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -16.5918007, 34.7164307, -45.7679825, 77.4062424, -93.9980469, 80.4844131
1: -18.3224945, 28.1499214, -50.0280113, 65.5433350, -83.8657990, 78.1779099
2: -18.7897148, 27.8247471, -51.2725220, 65.7026978, -84.4924011, 79.0972595
3: -22.1792717, 32.4609222, -59.0105782, 75.9060974, -98.0853729, 91.4714966
4: -21.7783241, 31.6278801, -53.9547539, 76.1816177, -97.9599304, 85.5826340

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B2_B2_A2_A1_A1

### Relational analysis result of IS_B2_A1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5128462, upper bound: 96.5829966
time: 1.05 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_A1_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5147836, upper bound: 96.5833005
time: 0.88 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -21.5673180, 41.9855995, -45.7679825, 77.4062424, -98.9735565, 87.7535706
1: -23.6898689, 34.6065483, -50.0280113, 65.5433350, -89.2331924, 84.6345444
2: -24.3597431, 34.2851677, -51.2725220, 65.7026978, -90.0624390, 85.5576782
3: -28.4248352, 40.0198708, -59.0105782, 75.9060974, -104.3309326, 99.0304489
4: -27.4042778, 39.1560936, -53.9547539, 76.1816177, -103.5858917, 93.1108475

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B2_B2_A2_A2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5128462, upper bound: 96.5908823
time: 1.09 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_A2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5147836, upper bound: 96.5861616
time: 0.95 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.32 seconds
IS_B1_A2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5785689, upper bound: 96.4956574
IS_B1_A2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5785689, upper bound: 96.4956574
IS_B1_A2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5788595, upper bound: 96.4976957
IS_B1_A2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5788595, upper bound: 96.4976957
IS_B1_A2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5829966, upper bound: 96.5135999
IS_B1_A2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5833005, upper bound: 96.5155373
IS_B1_A2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5829966, upper bound: 96.5220685
IS_B1_A2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5833005, upper bound: 96.5221098
IS_B2_A1_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.3884991, upper bound: 96.5293527
IS_B2_A1_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4822124, upper bound: 96.5674864
IS_B2_A1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.3884991, upper bound: 96.5519902
IS_B2_A1_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4822124, upper bound: 96.5674864
IS_B2_A1_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5751599
IS_B2_A1_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5058461, upper bound: 96.5736130
IS_B2_A1_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5819965
IS_B2_A1_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5058461, upper bound: 96.5736130
IS_B2_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4404351, upper bound: 96.5482518
IS_B2_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4810937, upper bound: 96.5535912
IS_B2_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4980814, upper bound: 96.5665094
IS_B2_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4980814, upper bound: 96.5661032
IS_B2_A1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5785852
IS_B2_A1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5058461, upper bound: 96.5770383
IS_B2_A1_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5734259
IS_B2_A1_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5058461, upper bound: 96.5738318
IS_B2_A1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5772027
IS_B2_A1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5772027
IS_B2_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5774933
IS_B2_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5774933
IS_B2_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4584912, upper bound: 96.5421771
IS_B2_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5041230, upper bound: 96.5504685
IS_B2_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5043528, upper bound: 96.5555867
IS_B2_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5043528, upper bound: 96.5528129
IS_B2_A1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5785689
IS_B2_A1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5782260
IS_B2_A1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5788595
IS_B2_A1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5782398
IS_B2_A1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5128462, upper bound: 96.5829966
IS_B2_A1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5147836, upper bound: 96.5833005
IS_B2_A1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5128462, upper bound: 96.5908823
IS_B2_A1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.32
Output dim: 4, lower bound: -96.5147836, upper bound: 96.5861616

## BFS IS instance: IS_B1_A2_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -45.7679825, 77.4062424, -12.6529942, 28.9299278, -74.6978989, 90.0592346
1: -50.0280113, 65.5433350, -14.1389532, 23.2849216, -73.3129349, 79.6822815
2: -51.2725220, 65.7026978, -14.4914856, 22.9595032, -74.2320099, 80.1941681
3: -59.0105782, 75.9060974, -17.4013557, 26.8606377, -85.8712158, 93.3074493
4: -53.9547539, 76.1816177, -17.6748047, 25.9647102, -79.9194565, 93.8564224

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5773154, upper bound: 96.4956443
time: 1.06 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5326399, upper bound: 96.3937417
time: 1.35 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B1_B2

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5776826, upper bound: 96.4912477
time: 1.03 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -45.7679825, 77.4062424, -17.9061203, 36.3579025, -82.1258850, 95.3123627
1: -50.0280113, 65.5433350, -19.7631435, 29.8415222, -79.8695221, 85.3064804
2: -51.2725220, 65.7026978, -20.3178749, 29.5494518, -80.8219757, 86.0205612
3: -59.0105782, 75.9060974, -23.8853111, 34.5401344, -93.5507126, 99.7914124
4: -53.9547539, 76.1816177, -23.4180298, 33.6300697, -87.5848083, 99.5996475

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5773154, upper bound: 96.4956443
time: 0.86 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5326399, upper bound: 96.3937417
time: 1.18 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5776826, upper bound: 96.4912477
time: 0.95 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -45.7656784, 77.4034882, -12.7826777, 29.1857872, -74.9514618, 90.1861649
1: -50.0255280, 65.5407181, -14.2728367, 23.4263058, -73.4518356, 79.8135529
2: -51.2699890, 65.7000122, -14.6236591, 23.1075191, -74.3775101, 80.3236618
3: -59.0077667, 75.9030762, -17.5459633, 27.0184841, -86.0262299, 93.4490356
4: -53.9522552, 76.1784439, -17.7854691, 26.1536980, -80.1059418, 93.9639053

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5776059, upper bound: 96.4976826
time: 0.79 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5345421, upper bound: 96.3986977
time: 0.77 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B1_B2

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5779680, upper bound: 96.4935835
time: 0.90 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -45.7656784, 77.4034882, -17.8593445, 36.4562531, -82.2219315, 95.2628098
1: -50.0255280, 65.5407181, -19.7178116, 29.8436108, -79.8691254, 85.2585297
2: -51.2699890, 65.7000122, -20.2700310, 29.5461750, -80.8161621, 85.9700241
3: -59.0077667, 75.9030762, -23.8437710, 34.5360870, -93.5438232, 99.7468338
4: -53.9522552, 76.1784439, -23.3809910, 33.6383438, -87.5905762, 99.5594254

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5776060, upper bound: 96.4976826
time: 1.00 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5404091, upper bound: 96.4003550
time: 1.24 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5755456, upper bound: 96.4807379
time: 0.91 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -45.7679825, 77.4062424, -16.3581161, 34.5058670, -80.2738495, 93.7643585
1: -50.0280113, 65.5433350, -18.0851593, 27.9535885, -77.9815979, 83.6284943
2: -51.2725220, 65.7026978, -18.5455475, 27.6234665, -78.8959808, 84.2482300
3: -59.0105782, 75.9060974, -21.9328194, 32.2270279, -91.2376099, 97.8389130
4: -53.9547539, 76.1816177, -21.5703163, 31.3907204, -85.3454742, 97.7519379

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5817430, upper bound: 96.5135868
time: 0.86 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5829272, upper bound: 96.5135999
time: 0.79 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B1_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5821123, upper bound: 96.5107135
time: 1.38 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -45.7656784, 77.4034882, -16.4992142, 34.7817383, -80.5474091, 93.9026947
1: -50.0255280, 65.5407181, -18.2325459, 28.1183300, -78.1438446, 83.7732544
2: -51.2699890, 65.7000122, -18.6936512, 27.7920303, -79.0620193, 84.3936615
3: -59.0077667, 75.9030762, -22.0940952, 32.4168167, -91.4245682, 97.9971695
4: -53.9522552, 76.1784439, -21.7062378, 31.6019878, -85.5542297, 97.8846741

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5820469, upper bound: 96.5155242
time: 1.04 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5832263, upper bound: 96.5155373
time: 1.08 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_B1_B2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5822834, upper bound: 96.5124717
time: 1.00 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -45.7679825, 77.4062424, -21.3495865, 41.7910309, -87.5590134, 98.7558289
1: -50.0280113, 65.5433350, -23.4670982, 34.4257469, -84.4537582, 89.0104294
2: -51.2725220, 65.7026978, -24.1332436, 34.0979691, -85.3704910, 89.8359375
3: -59.0105782, 75.9060974, -28.1921997, 39.8060646, -98.8166428, 104.0982971
4: -53.9547539, 76.1816177, -27.2068214, 38.9321404, -92.8868866, 103.3884277

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5848491, upper bound: 96.5220529
time: 1.33 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5857493, upper bound: 96.5202683
time: 0.94 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B1_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5817330, upper bound: 96.5048881
time: 1.13 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -45.7656784, 77.4034882, -21.4111347, 41.9651108, -87.7307892, 98.8146210
1: -50.0255280, 65.5407181, -23.5329018, 34.4902267, -84.5157394, 89.0736237
2: -51.2699890, 65.7000122, -24.1938686, 34.1634369, -85.4334259, 89.8938751
3: -59.0077667, 75.9030762, -28.2649994, 39.8746490, -98.8824005, 104.1680756
4: -53.9522552, 76.1784439, -27.2619457, 39.0278549, -92.9801102, 103.4403839

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5848550, upper bound: 96.5220942
time: 0.94 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5857557, upper bound: 96.5203054
time: 1.40 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_B2_B2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5795275, upper bound: 96.5032805
time: 1.02 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -8.7550936, 23.5118904, -40.2649155, 68.9257889, -77.6808853, 63.7768059
1: -10.0049467, 18.6291389, -43.9982300, 57.9924126, -67.9973526, 62.6273613
2: -10.2514791, 18.3245850, -45.0749283, 58.0296440, -68.2811050, 63.3995132
3: -12.6735754, 21.4338436, -51.8495407, 67.1525803, -79.8261490, 73.2833862
4: -13.6229401, 20.5587692, -47.7031250, 67.1675110, -80.7904510, 68.2618942

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.3882417, upper bound: 96.5292472
time: 1.05 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.3834782, upper bound: 96.5283228
time: 0.90 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.3866392, upper bound: 96.5290881
time: 1.03 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.3845218, upper bound: 96.5275163
time: 0.95 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 34
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 29

Time for candidate selection: 18.76 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 46

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.3884991, upper bound: 96.5293527
time: 0.98 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.3634604, upper bound: 96.5260813
time: 0.97 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -12.5336199, 28.6829758, -40.8853493, 69.7066422, -82.2402649, 69.5683212
1: -13.9861374, 23.0520210, -44.6765060, 58.7597847, -72.7459106, 67.7285309
2: -14.3491745, 22.7370377, -45.7650757, 58.7995110, -73.1486816, 68.5021133
3: -17.2136307, 26.5880470, -52.6343346, 68.0640945, -85.2777176, 79.2223511
4: -17.4574356, 25.7014580, -48.4212799, 68.0924225, -85.5498581, 74.1227417

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4889565, upper bound: 96.5663449
time: 1.01 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4374794, upper bound: 96.5384232
time: 0.77 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4557163, upper bound: 96.5549066
time: 1.03 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A1_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4843405, upper bound: 96.5675145
time: 1.06 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -14.2849388, 30.7818317, -40.2649155, 68.9257889, -83.2107239, 71.0467453
1: -15.8485060, 25.0549927, -43.9982300, 57.9924126, -73.8409119, 69.0532227
2: -16.2596607, 24.7564201, -45.0749283, 58.0296440, -74.2892914, 69.8313446
3: -19.2824383, 28.9951706, -51.8495407, 67.1525803, -86.4349976, 80.8447113
4: -19.3188915, 28.0527802, -47.7031250, 67.1675110, -86.4863892, 75.7559052

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4378875, upper bound: 96.5504341
time: 0.85 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4289781, upper bound: 96.5321481
time: 1.01 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4378279, upper bound: 96.5516692
time: 1.03 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4362341, upper bound: 96.5517524
time: 0.79 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4339428, upper bound: 96.5515128
time: 1.01 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4379888, upper bound: 96.5519902
time: 0.94 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A1_B2

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4355592, upper bound: 96.5444091
time: 0.98 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -17.2680473, 35.5672455, -40.8853493, 69.7066422, -86.9746857, 76.4525909
1: -19.0706825, 29.0906296, -44.6765060, 58.7597847, -77.8304520, 73.7671356
2: -19.6311646, 28.7999249, -45.7650757, 58.7995110, -78.4306793, 74.5650024
3: -23.1020317, 33.6565971, -52.6343346, 68.0640945, -91.1661224, 86.2909164
4: -22.7006683, 32.7355652, -48.4212799, 68.0924225, -90.7930908, 81.1568375

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4821941, upper bound: 96.5653939
time: 0.93 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4822124, upper bound: 96.5674176
time: 0.91 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4821889, upper bound: 96.5609291
time: 0.86 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4813999, upper bound: 96.5674853
time: 1.07 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4788071, upper bound: 96.5671949
time: 1.05 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4582197, upper bound: 96.5612180
time: 1.30 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -15.6721573, 33.5140343, -40.8853493, 69.7066422, -85.3787994, 74.3993835
1: -17.3391151, 27.0493793, -44.6765060, 58.7597847, -76.0988846, 71.7258835
2: -17.7767715, 26.7265797, -45.7650757, 58.7995110, -76.5762787, 72.4916534
3: -21.0434837, 31.1638126, -52.6343346, 68.0640945, -89.1075745, 83.7981339
4: -20.8186436, 30.3296700, -48.4212799, 68.0924225, -88.9110641, 78.7509384

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164251, upper bound: 96.5730675
time: 0.81 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164199, upper bound: 96.5686198
time: 0.92 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_B1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5164030, upper bound: 96.5751599
time: 0.71 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A1_B2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5109952, upper bound: 96.5585619
time: 0.90 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -18.9047394, 38.1531067, -40.8502731, 69.6557236, -88.5604630, 79.0033798
1: -20.8723717, 31.5507431, -44.6382599, 58.7163734, -79.5887299, 76.1890030
2: -21.4519386, 31.1929474, -45.7262497, 58.7556076, -80.2075272, 76.9191895
3: -25.2917919, 36.4526367, -52.5902634, 68.0134888, -93.3052826, 89.0428848
4: -24.6734753, 35.5904541, -48.3817368, 68.0396042, -92.7130585, 83.9721909

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5133883, upper bound: 96.5721144
time: 0.97 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5133831, upper bound: 96.5676588
time: 0.76 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5114520, upper bound: 96.5738709
time: 1.18 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A1_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5089638, upper bound: 96.5731615
time: 1.17 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -20.6747246, 40.7641144, -40.8853493, 69.7066422, -90.3813629, 81.6494446
1: -22.7270298, 33.4905243, -44.6765060, 58.7597847, -81.4867935, 78.1670303
2: -23.3768768, 33.1677475, -45.7650757, 58.7995110, -82.1763916, 78.9328156
3: -27.3090744, 38.7219353, -52.6343346, 68.0640945, -95.3731689, 91.3562469
4: -26.4379921, 37.8381767, -48.4212799, 68.0924225, -94.5304108, 86.2594376

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4654691, upper bound: 96.5680727
time: 1.10 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5122071, upper bound: 96.5794620
time: 0.92 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -24.4922638, 45.6690979, -40.8502731, 69.6557236, -94.1479874, 86.5193634
1: -26.8402290, 38.2452202, -44.6382599, 58.7163734, -85.5565948, 82.8834839
2: -27.5692806, 37.9812622, -45.7262497, 58.7556076, -86.3248672, 83.7074890
3: -32.0349541, 44.2854080, -52.5902634, 68.0134888, -100.0484467, 96.8756638
4: -30.6520195, 43.5106010, -48.3817368, 68.0396042, -98.6916199, 91.8923340

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5058278, upper bound: 96.5715205
time: 1.33 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5058226, upper bound: 96.5670750
time: 1.16 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5028456, upper bound: 96.5732725
time: 0.76 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5018114, upper bound: 96.5712269
time: 1.05 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.9277182, 31.8642063, -43.9486198, 75.3157043, -90.2434158, 75.8128052
1: -16.5565453, 26.0010738, -48.0827293, 63.6439819, -80.2005310, 74.0838013
2: -16.9849052, 25.6875801, -49.2626686, 63.7824173, -80.7673035, 74.9502335
3: -20.1307087, 30.1114693, -56.7959290, 73.6697388, -93.8004456, 86.9073944
4: -20.1693687, 29.1383858, -51.9276237, 73.8320770, -94.0014496, 81.0660095

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4404351, upper bound: 96.5482518
time: 0.90 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4280056, upper bound: 96.5380725
time: 0.94 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4280056, upper bound: 96.5482518
time: 0.83 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.90 seconds
IS_B1_A2_A2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5326399, upper bound: 96.3937417
IS_B1_A2_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5776826, upper bound: 96.4912477
IS_B1_A2_A2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5326399, upper bound: 96.3937417
IS_B1_A2_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5776826, upper bound: 96.4912477
IS_B1_A2_A2_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5345421, upper bound: 96.3986977
IS_B1_A2_A2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5779680, upper bound: 96.4935835
IS_B1_A2_A2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5404091, upper bound: 96.4003550
IS_B1_A2_A2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5755456, upper bound: 96.4807379
IS_B1_A2_A2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5829272, upper bound: 96.5135999
IS_B1_A2_A2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5821123, upper bound: 96.5107135
IS_B1_A2_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5832263, upper bound: 96.5155373
IS_B1_A2_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5822834, upper bound: 96.5124717
IS_B1_A2_A2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5857493, upper bound: 96.5202683
IS_B1_A2_A2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5817330, upper bound: 96.5048881
IS_B1_A2_A2_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5857557, upper bound: 96.5203054
IS_B1_A2_A2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5795275, upper bound: 96.5032805
IS_B2_A1_B1_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.3884991, upper bound: 96.5293527
IS_B2_A1_B1_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.3634604, upper bound: 96.5260813
IS_B2_A1_B1_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.4557163, upper bound: 96.5549066
IS_B2_A1_B1_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.4843405, upper bound: 96.5675145
IS_B2_A1_B1_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.4379888, upper bound: 96.5519902
IS_B2_A1_B1_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.4355592, upper bound: 96.5444091
IS_B2_A1_B1_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.4788071, upper bound: 96.5671949
IS_B2_A1_B1_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.4582197, upper bound: 96.5612180
IS_B2_A1_B1_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5164030, upper bound: 96.5751599
IS_B2_A1_B1_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5109952, upper bound: 96.5585619
IS_B2_A1_B1_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5114520, upper bound: 96.5738709
IS_B2_A1_B1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5089638, upper bound: 96.5731615
IS_B2_A1_B1_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.4654691, upper bound: 96.5680727
IS_B2_A1_B1_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5122071, upper bound: 96.5794620
IS_B2_A1_B1_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5028456, upper bound: 96.5732725
IS_B2_A1_B1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.5018114, upper bound: 96.5712269
IS_B2_A1_B1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.4280056, upper bound: 96.5380725
IS_B2_A1_B1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 4, lower bound: -96.4280056, upper bound: 96.5482518
IS_B2_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.4810937, upper bound: 96.5535912
IS_B2_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.4980814, upper bound: 96.5665094
IS_B2_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.4980814, upper bound: 96.5661032
IS_B2_A1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5785852
IS_B2_A1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.5058461, upper bound: 96.5770383
IS_B2_A1_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.5164434, upper bound: 96.5734259
IS_B2_A1_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.5058461, upper bound: 96.5738318
IS_B2_A1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5772027
IS_B2_A1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5772027
IS_B2_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5774933
IS_B2_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5774933
IS_B2_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.4584912, upper bound: 96.5421771
IS_B2_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.5041230, upper bound: 96.5504685
IS_B2_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.5043528, upper bound: 96.5555867
IS_B2_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.5043528, upper bound: 96.5528129
IS_B2_A1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5785689
IS_B2_A1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.4949037, upper bound: 96.5782260
IS_B2_A1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5788595
IS_B2_A1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.4969419, upper bound: 96.5782398
IS_B2_A1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.5128462, upper bound: 96.5829966
IS_B2_A1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.5147836, upper bound: 96.5833005
IS_B2_A1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.5128462, upper bound: 96.5908823
IS_B2_A1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.90
Output dim: 4, lower bound: -96.5147836, upper bound: 96.5861616
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=108.20734405517578
rel_dist={4: [-96.61858128162753, 96.61858128162754]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5919846, upper bound: 96.5299857
time: 0.68 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6152409, upper bound: 96.6152408
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.63 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 4, lower bound: -96.5919846, upper bound: 96.5299857
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 4, lower bound: -96.6152409, upper bound: 96.6152408

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -33.7610512, 56.9028625, -23.6491947, 44.5811996, -78.3422546, 80.5520554
1: -36.8422394, 48.6217384, -25.9020424, 36.9293671, -73.7716064, 74.5237732
2: -37.7350235, 48.5267754, -26.6062851, 36.6187668, -74.3537674, 75.1330566
3: -43.3244705, 56.4820633, -30.8861237, 42.7392807, -86.0637512, 87.3681793
4: -40.1423340, 56.1176567, -29.5871983, 41.9198265, -82.0621567, 85.7048340

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5841542, upper bound: 96.5289622
time: 0.89 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5841542, upper bound: 96.5297792
time: 0.96 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -37.1937485, 61.3177948, -47.7588348, 79.9911652, -117.1848907, 109.0766296
1: -40.5940704, 53.0116119, -52.1639366, 67.9121017, -108.5061722, 105.1755524
2: -41.5752144, 52.9253693, -53.4510193, 68.0761642, -109.6513748, 106.3763885
3: -47.6881599, 61.5470695, -61.4133224, 78.7076035, -126.3957596, 122.9603882
4: -44.0563965, 61.3981972, -56.2418251, 79.0030441, -123.0594406, 117.6400223

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6143514, upper bound: 96.6084821
time: 0.96 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6150213, upper bound: 96.6150204
time: 1.01 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.07 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 4.07
Output dim: 4, lower bound: -96.5841542, upper bound: 96.5289622
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 4.07
Output dim: 4, lower bound: -96.5841542, upper bound: 96.5297792
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 4.07
Output dim: 4, lower bound: -96.6143514, upper bound: 96.6084821
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 4.07
Output dim: 4, lower bound: -96.6150213, upper bound: 96.6150204

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -33.1401825, 56.3221970, -23.6326866, 44.5647583, -77.7049179, 79.9548798
1: -36.1802673, 48.0019684, -25.8849945, 36.9137573, -73.0940247, 73.8869629
2: -37.0649910, 47.8857155, -26.5887222, 36.6027679, -73.6677551, 74.4744186
3: -42.5984001, 55.7483406, -30.8680954, 42.7208328, -85.3192291, 86.6164246
4: -39.4894714, 55.3545036, -29.5702343, 41.9010048, -81.3904572, 84.9247360

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4764256, upper bound: 96.5117448
time: 0.81 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4764256, upper bound: 96.5289622
time: 0.75 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -33.7217102, 56.8639793, -23.6491947, 44.5811996, -78.3029022, 80.5131607
1: -36.8003616, 48.5846863, -25.9020424, 36.9293671, -73.7297211, 74.4867096
2: -37.6931419, 48.4877625, -26.6062851, 36.6187668, -74.3118744, 75.0940475
3: -43.2789116, 56.4380875, -30.8861237, 42.7392807, -86.0181885, 87.3242035
4: -40.1029587, 56.0702896, -29.5871983, 41.9198265, -82.0227814, 85.6574860

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5178301, upper bound: 96.5178312
time: 0.87 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5178301, upper bound: 96.5297792
time: 1.07 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -37.1743088, 61.2958984, -47.1010628, 79.3604965, -116.5348053, 108.3969498
1: -40.5732269, 52.9911079, -51.4631310, 67.2346344, -107.8078613, 104.4542389
2: -41.5541382, 52.9043846, -52.7382889, 67.3822784, -108.9364166, 105.6426697
3: -47.6650467, 61.5229874, -60.6423721, 77.9089737, -125.5740204, 122.1653595
4: -44.0355148, 61.3729515, -55.5394173, 78.1715088, -122.2070236, 116.9123688

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6078122, upper bound: 96.6078122
time: 0.89 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6078122, upper bound: 96.6078122
time: 1.27 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -37.1937485, 61.3177948, -47.7190552, 79.9503479, -117.1440811, 109.0368500
1: -40.5940704, 53.0116119, -52.1214371, 67.8727264, -108.4667892, 105.1330414
2: -41.5752144, 52.9253693, -53.4082642, 68.0354233, -109.6106415, 106.3336334
3: -47.6881599, 61.5470695, -61.3669090, 78.6607513, -126.3488998, 122.9139786
4: -44.0563965, 61.3981972, -56.2001648, 78.9533081, -123.0097046, 117.5983582

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1

### Relational analysis result of IS_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6091724, upper bound: 96.5942028
time: 1.00 seconds

## Relational analysis of IS_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6110830, upper bound: 96.6110820
time: 0.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.10 seconds
IS_B1_A1_A1, status: Status.VERIFIED, split count: 3, time: 4.10
Output dim: 4, lower bound: -96.4764256, upper bound: 96.5117448
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 4, lower bound: -96.4764256, upper bound: 96.5289622
IS_B1_A2_A1, status: Status.VERIFIED, split count: 3, time: 4.10
Output dim: 4, lower bound: -96.5178301, upper bound: 96.5178312
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 4, lower bound: -96.5178301, upper bound: 96.5297792
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 4, lower bound: -96.6078122, upper bound: 96.6078122
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 4, lower bound: -96.6078122, upper bound: 96.6078122
IS_B2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 4, lower bound: -96.6091724, upper bound: 96.5942028
IS_B2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 4, lower bound: -96.6110830, upper bound: 96.6110820

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -46.7805595, 78.7027283, -23.6326866, 44.5647583, -91.3453064, 102.3354111
1: -51.1058922, 66.6815567, -25.8849945, 36.9137573, -88.0196533, 92.5665512
2: -52.3633614, 66.8287354, -26.5887222, 36.6027679, -88.9661255, 93.4174576
3: -60.2011070, 77.2774200, -30.8680954, 42.7208328, -102.9219284, 108.1455078
4: -55.1545639, 77.5463715, -29.5702343, 41.9010048, -97.0555420, 107.1166077

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4468426, upper bound: 96.4611855
time: 0.80 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4541045, upper bound: 96.5160596
time: 0.80 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -47.5846710, 79.7620087, -23.6491947, 44.5811996, -92.1658707, 103.4112015
1: -51.9733887, 67.6882935, -25.9020424, 36.9293671, -88.9027557, 93.5903091
2: -53.2536049, 67.8520279, -26.6062851, 36.6187668, -89.8723450, 94.4583130
3: -61.1927147, 78.4438705, -30.8861237, 42.7392807, -103.9319916, 109.3299942
4: -56.0425453, 78.7357559, -29.5871983, 41.9198265, -97.9623642, 108.3229523

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5058356, upper bound: 96.5014966
time: 0.91 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5176477, upper bound: 96.5176488
time: 1.04 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -36.5598450, 60.7141609, -47.1010628, 79.3604965, -115.9203415, 107.8152237
1: -39.9176331, 52.3708458, -51.4631310, 67.2346344, -107.1522675, 103.8339767
2: -40.8898468, 52.2646866, -52.7382889, 67.3822784, -108.2721252, 105.0029755
3: -46.9441566, 60.7913551, -60.6423721, 77.9089737, -124.8531342, 121.4337311
4: -43.3838768, 60.6105232, -55.5394173, 78.1715088, -121.5553894, 116.1499405

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_B1

### Relational analysis result of IS_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6015330, upper bound: 96.5899375
time: 1.09 seconds

## Relational analysis of IS_B2_B1_A1_B2

### Relational analysis result of IS_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6029379, upper bound: 96.6029378
time: 0.95 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -37.1532173, 61.2768593, -47.1010628, 79.3604965, -116.5137177, 108.3778992
1: -40.5508957, 52.9722595, -51.4631310, 67.2346344, -107.7855072, 104.4353943
2: -41.5317917, 52.8844566, -52.7382889, 67.3822784, -108.9140472, 105.6227417
3: -47.6410065, 61.5009003, -60.6423721, 77.9089737, -125.5499725, 122.1432724
4: -44.0144424, 61.3488579, -55.5394173, 78.1715088, -122.1859512, 116.8882675

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A2_B1

### Relational analysis result of IS_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6015330, upper bound: 96.5899375
time: 0.93 seconds

## Relational analysis of IS_B2_B1_A2_B2

### Relational analysis result of IS_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6029379, upper bound: 96.6029378
time: 0.81 seconds

## BFS IS instance: IS_B2_B2_B1

### Backsubstitution after applying IS history:
0: -36.6290588, 60.5479050, -46.7894211, 78.8589554, -115.4879761, 107.3373260
1: -39.9900818, 52.3307648, -51.1519814, 67.0058212, -106.9958878, 103.4827423
2: -40.9534874, 52.2398224, -52.3858948, 67.1554642, -108.1089478, 104.6257172
3: -46.9992752, 60.7510872, -60.2768440, 77.6478806, -124.6471176, 121.0279160
4: -43.4164047, 60.5800705, -55.1636429, 77.8824539, -121.2988586, 115.7437134

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_B1_A1

### Relational analysis result of IS_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6015285, upper bound: 96.5935390
time: 1.30 seconds

## Relational analysis of IS_B2_B2_B1_A2

### Relational analysis result of IS_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6015285, upper bound: 96.5899375
time: 1.00 seconds

## BFS IS instance: IS_B2_B2_B2

### Backsubstitution after applying IS history:
0: -36.8585320, 60.8513184, -46.0311928, 77.7005310, -114.5590363, 106.8824921
1: -40.2359695, 52.5975456, -50.3242874, 65.8815689, -106.1175385, 102.9218292
2: -41.2073364, 52.5052948, -51.5641251, 66.0073242, -107.2146606, 104.0694122
3: -47.2790604, 61.0627747, -59.3308029, 76.3739243, -123.6529541, 120.3935623
4: -43.6835938, 60.9064941, -54.3605728, 76.5541840, -120.2377777, 115.2670593

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B2_B1

### Relational analysis result of IS_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979978, upper bound: 96.5987888
time: 0.88 seconds

## Relational analysis of IS_B2_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5986621, upper bound: 96.5986609
time: 1.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.17 seconds
IS_B1_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.17
Output dim: 4, lower bound: -96.4468426, upper bound: 96.4611855
IS_B1_A1_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.17
Output dim: 4, lower bound: -96.4541045, upper bound: 96.5160596
IS_B1_A2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.17
Output dim: 4, lower bound: -96.5058356, upper bound: 96.5014966
IS_B1_A2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.17
Output dim: 4, lower bound: -96.5176477, upper bound: 96.5176488
IS_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -96.6015330, upper bound: 96.5899375
IS_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -96.6029379, upper bound: 96.6029378
IS_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -96.6015330, upper bound: 96.5899375
IS_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -96.6029379, upper bound: 96.6029378
IS_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -96.6015285, upper bound: 96.5935390
IS_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -96.6015285, upper bound: 96.5899375
IS_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -96.5979978, upper bound: 96.5987888
IS_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.17
Output dim: 4, lower bound: -96.5986621, upper bound: 96.5986609

## BFS IS instance: IS_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -35.9907494, 59.9387436, -46.2494240, 78.3527451, -114.3434906, 106.1881714
1: -39.3093796, 51.6849442, -50.5749245, 66.4542007, -105.7635651, 102.2598724
2: -40.2634239, 51.5738029, -51.7997055, 66.5883789, -106.8518066, 103.3735046
3: -46.2483597, 59.9897041, -59.6375389, 76.9996033, -123.2479553, 119.6272125
4: -42.7418518, 59.7857857, -54.5916595, 77.2020187, -119.9438477, 114.3774414

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A1_B1_A1

### Relational analysis result of IS_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5498342, upper bound: 96.5782974
time: 0.85 seconds

## Relational analysis of IS_B2_B1_A1_B1_A2

### Relational analysis result of IS_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6007334, upper bound: 96.5885607
time: 0.85 seconds

## BFS IS instance: IS_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -36.2286415, 60.2509766, -45.4270706, 77.1179504, -113.3465881, 105.6780396
1: -39.5642052, 51.9602051, -49.6798515, 65.2524567, -104.8166656, 101.6400604
2: -40.5262222, 51.8483658, -50.9078789, 65.3638458, -105.8900604, 102.7562408
3: -46.5387802, 60.3112564, -58.6183395, 75.6323090, -122.1710815, 118.9295883
4: -43.0157089, 60.1233521, -53.7122421, 75.7832260, -118.7989349, 113.8355865

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B2_B1

### Relational analysis result of IS_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5892777, upper bound: 96.5867962
time: 0.99 seconds

## Relational analysis of IS_B2_B1_A1_B2_B2

### Relational analysis result of IS_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5900988, upper bound: 96.5900986
time: 1.32 seconds

## BFS IS instance: IS_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -36.5885849, 60.5068550, -46.2494240, 78.3527451, -114.9413300, 106.7562790
1: -39.9469299, 52.2913551, -50.5749245, 66.4542007, -106.4011078, 102.8662720
2: -40.9101219, 52.1988335, -51.7997055, 66.5883789, -107.4984970, 103.9985352
3: -46.9521179, 60.7048607, -59.6375389, 76.9996033, -123.9517212, 120.3423691
4: -43.3744926, 60.5306206, -54.5916595, 77.2020187, -120.5765076, 115.1222839

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A2_B1_A1

### Relational analysis result of IS_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5117431, upper bound: 96.5768431
time: 0.91 seconds

## Relational analysis of IS_B2_B1_A2_B1_A2

### Relational analysis result of IS_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5117431, upper bound: 96.5906010
time: 0.96 seconds

## BFS IS instance: IS_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -36.8180161, 60.8104172, -45.4270706, 77.1179504, -113.9359589, 106.2374878
1: -40.1927338, 52.5582047, -49.6798515, 65.2524567, -105.4451904, 102.2380524
2: -41.1639328, 52.4644241, -50.9078789, 65.3638458, -106.5277786, 103.3722992
3: -47.2317886, 61.0166397, -58.6183395, 75.6323090, -122.8640976, 119.6349792
4: -43.6417236, 60.8572044, -53.7122421, 75.7832260, -119.4249496, 114.5694275

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A2_B2_B1

### Relational analysis result of IS_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5969763, upper bound: 96.5876571
time: 1.01 seconds

## Relational analysis of IS_B2_B1_A2_B2_B2

### Relational analysis result of IS_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977992, upper bound: 96.5909436
time: 1.08 seconds

## BFS IS instance: IS_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -35.9907494, 59.9387436, -46.7894211, 78.8589554, -114.8496857, 106.7281570
1: -39.3093796, 51.6849442, -51.1519814, 67.0058212, -106.3151932, 102.8369141
2: -40.2634239, 51.5738029, -52.3858948, 67.1554642, -107.4188843, 103.9596863
3: -46.2483597, 59.9897041, -60.2768440, 77.6478806, -123.8962173, 120.2665176
4: -42.7418518, 59.7857857, -55.1636429, 77.8824539, -120.6242981, 114.9494324

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A1_A1

### Relational analysis result of IS_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5885629, upper bound: 96.5885627
time: 0.86 seconds

## Relational analysis of IS_B2_B2_B1_A1_A2

### Relational analysis result of IS_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5885629, upper bound: 96.5935390
time: 0.95 seconds

## BFS IS instance: IS_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -36.5885849, 60.5068550, -46.7894211, 78.8589554, -115.4475250, 107.2962646
1: -39.9469299, 52.2913551, -51.1519814, 67.0058212, -106.9527359, 103.4433365
2: -40.9101219, 52.1988335, -52.3858948, 67.1554642, -108.0655823, 104.5847321
3: -46.9521179, 60.7048607, -60.2768440, 77.6478806, -124.5999756, 120.9816971
4: -43.3744926, 60.5306206, -55.1636429, 77.8824539, -121.2569427, 115.6942596

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_B1_A2_A1

### Relational analysis result of IS_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5498290, upper bound: 96.5782974
time: 1.00 seconds

## Relational analysis of IS_B2_B2_B1_A2_A2

### Relational analysis result of IS_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6007275, upper bound: 96.5885607
time: 1.22 seconds

## BFS IS instance: IS_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -34.8043900, 57.9407539, -40.1067276, 68.4115601, -103.2159424, 98.0474854
1: -38.0166054, 50.0375404, -43.8479156, 57.7339668, -95.7505722, 93.8854523
2: -38.9476166, 49.9189491, -44.9056244, 57.7641373, -96.7117538, 94.8245697
3: -44.7388687, 58.0796356, -51.6687050, 66.8986969, -111.6375656, 109.7483368
4: -41.3881836, 57.8689194, -47.5603218, 66.8738327, -108.2620163, 105.4292374

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B2_B1_A1

### Relational analysis result of IS_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5139443, upper bound: 96.5801659
time: 1.04 seconds

## Relational analysis of IS_B2_B2_B2_B1_A2

### Relational analysis result of IS_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5139443, upper bound: 96.5987888
time: 0.83 seconds

## BFS IS instance: IS_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -36.5685501, 60.4561272, -44.0557594, 75.1185532, -111.6870956, 104.5118866
1: -39.9242249, 52.2416878, -48.2045212, 63.4997864, -103.4239960, 100.4462128
2: -40.8895798, 52.1495514, -49.3970032, 63.6265450, -104.5161285, 101.5465546
3: -46.9264221, 60.6477013, -56.9382095, 73.5630112, -120.4894257, 117.5858994
4: -43.3508835, 60.4850616, -52.0773964, 73.7352829, -117.0861511, 112.5624542

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_B2_B2_A1

### Relational analysis result of IS_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5591758, upper bound: 96.5898017
time: 0.99 seconds

## Relational analysis of IS_B2_B2_B2_B2_A2

### Relational analysis result of IS_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5982064, upper bound: 96.5982053
time: 0.95 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.19 seconds
IS_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5498342, upper bound: 96.5782974
IS_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.6007334, upper bound: 96.5885607
IS_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5892777, upper bound: 96.5867962
IS_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5900988, upper bound: 96.5900986
IS_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5117431, upper bound: 96.5768431
IS_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5117431, upper bound: 96.5906010
IS_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5969763, upper bound: 96.5876571
IS_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5977992, upper bound: 96.5909436
IS_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5885629, upper bound: 96.5885627
IS_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5885629, upper bound: 96.5935390
IS_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5498290, upper bound: 96.5782974
IS_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.6007275, upper bound: 96.5885607
IS_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5139443, upper bound: 96.5801659
IS_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5139443, upper bound: 96.5987888
IS_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5591758, upper bound: 96.5898017
IS_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.19
Output dim: 4, lower bound: -96.5982064, upper bound: 96.5982053

## BFS IS instance: IS_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -33.5072403, 56.8676605, -45.6192589, 77.5343170, -111.0415344, 102.4869080
1: -36.6332932, 48.7775650, -49.8882027, 65.6745453, -102.3078384, 98.6657639
2: -37.5541534, 48.6695595, -51.1045876, 65.8133011, -103.3674545, 99.7741470
3: -43.2543259, 56.5944824, -58.8566971, 76.0790482, -119.3333740, 115.4511642
4: -40.0165787, 56.3124504, -53.8708038, 76.2695541, -116.2861328, 110.1832428

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5498342, upper bound: 96.5782974
time: 0.98 seconds

## Relational analysis of IS_B2_B1_A1_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5498342, upper bound: 96.5782974
time: 0.90 seconds

## BFS IS instance: IS_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -35.1127472, 59.3864212, -45.7611084, 77.8297119, -112.9424591, 105.1475067
1: -38.3698540, 50.9215775, -50.0567780, 65.9632339, -104.3330841, 100.9783554
2: -39.3062553, 50.8122711, -51.2750854, 66.0913315, -105.3975830, 102.0873489
3: -45.3115692, 59.0991631, -59.0732384, 76.4241943, -121.7357635, 118.1724014
4: -41.7859917, 58.8542976, -54.0848274, 76.5993881, -118.3853607, 112.9391174

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_B1_A2_A1

### Relational analysis result of IS_B2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5868443, upper bound: 96.5868441
time: 0.83 seconds

## Relational analysis of IS_B2_B1_A1_B1_A2_A2

### Relational analysis result of IS_B2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5868443, upper bound: 96.5885607
time: 0.77 seconds

## BFS IS instance: IS_B2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -34.1840096, 57.3584862, -39.5154877, 67.8429031, -102.0269089, 96.8739777
1: -37.3569031, 49.4158630, -43.2165070, 57.1232910, -94.4801865, 92.6323700
2: -38.2777290, 49.2783165, -44.2634125, 57.1355820, -95.4133148, 93.5417328
3: -44.0136414, 57.3356247, -50.9715195, 66.1779785, -110.1916199, 108.3071365
4: -40.7323380, 57.1077919, -46.9210434, 66.1208649, -106.8532028, 104.0288391

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A1_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5470265, upper bound: 96.5765912
time: 0.98 seconds

## Relational analysis of IS_B2_B1_A1_B2_B1_A2

### Relational analysis result of IS_B2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5890115, upper bound: 96.5863209
time: 1.01 seconds

## BFS IS instance: IS_B2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -35.9412727, 59.8598175, -43.4514542, 74.5323486, -110.4736176, 103.3112717
1: -39.2554054, 51.6082916, -47.5596313, 62.8722229, -102.1276245, 99.1679153
2: -40.2114258, 51.4966888, -48.7404366, 62.9815750, -103.1929932, 100.2371216
3: -46.1893272, 59.9008675, -56.2256660, 72.8205719, -119.0098953, 116.1265335
4: -42.6863937, 59.7065125, -51.4321785, 72.9620056, -115.6483994, 111.1386871

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A1_B2_B2_A1

### Relational analysis result of IS_B2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5476609, upper bound: 96.5798941
time: 1.20 seconds

## Relational analysis of IS_B2_B1_A1_B2_B2_A2

### Relational analysis result of IS_B2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5896371, upper bound: 96.5896370
time: 1.05 seconds

## BFS IS instance: IS_B2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -23.0723076, 43.8761978, -46.2494240, 78.3527451, -101.4250412, 90.1256027
1: -25.2957878, 36.3010330, -50.5749245, 66.4542007, -91.7499847, 86.8759613
2: -25.9798889, 35.9876709, -51.7997055, 66.5883789, -92.5682678, 87.7873764
3: -30.2208290, 42.0018730, -59.6375389, 76.9996033, -107.2204208, 101.6394043
4: -28.9771233, 41.1698227, -54.5916595, 77.2020187, -106.1791382, 95.7614822

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A2_B1_A1_B1

### Relational analysis result of IS_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5040803, upper bound: 96.5503064
time: 0.99 seconds

## Relational analysis of IS_B2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5104767, upper bound: 96.5759977
time: 1.29 seconds

## Relational analysis of IS_B2_B1_A2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5104767, upper bound: 96.5768431
time: 1.03 seconds

## BFS IS instance: IS_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -47.1025543, 79.1225052, -46.2494240, 78.3527451, -125.4552841, 125.3719177
1: -51.4616356, 67.1144867, -50.5749245, 66.4542007, -117.9158096, 117.6894073
2: -52.7265778, 67.2794266, -51.7997055, 66.5883789, -119.3149567, 119.0791245
3: -60.6136703, 77.7703323, -59.6375389, 76.9996033, -137.6132507, 137.4078522
4: -55.4860535, 78.0511322, -54.5916595, 77.2020187, -132.6880798, 132.6427612

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5104767, upper bound: 96.5885629
time: 0.84 seconds

## Relational analysis of IS_B2_B1_A2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5104767, upper bound: 96.5906010
time: 0.86 seconds

## BFS IS instance: IS_B2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -34.7650337, 57.9014587, -39.5154877, 67.8429031, -102.6079102, 97.4169464
1: -37.9746323, 49.9998856, -43.2165070, 57.1232910, -95.0979156, 93.2163925
2: -38.9054260, 49.8796005, -44.2634125, 57.1355820, -96.0410080, 94.1430130
3: -44.6930923, 58.0346718, -50.9715195, 66.1779785, -110.8710632, 109.0061951
4: -41.3477783, 57.8216782, -46.9210434, 66.1208649, -107.4686432, 104.7427216

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A2_B2_B1_A1

### Relational analysis result of IS_B2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5576800, upper bound: 96.5788116
time: 1.04 seconds

## Relational analysis of IS_B2_B1_A2_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5967071, upper bound: 96.5871933
time: 1.06 seconds

## BFS IS instance: IS_B2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -36.5282059, 60.4154053, -43.4514542, 74.5323486, -111.0605545, 103.8668518
1: -39.8812332, 52.2025337, -47.5596313, 62.8722229, -102.7534561, 99.7621536
2: -40.8463402, 52.1088181, -48.7404366, 62.9815750, -103.8279114, 100.8492584
3: -46.8794556, 60.6017570, -56.2256660, 72.8205719, -119.7000275, 116.8274002
4: -43.3091316, 60.4359512, -51.4321785, 72.9620056, -116.2711182, 111.8681335

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A2_B2_B2_A1

### Relational analysis result of IS_B2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5583142, upper bound: 96.5820992
time: 1.02 seconds

## Relational analysis of IS_B2_B1_A2_B2_B2_A2

### Relational analysis result of IS_B2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5973327, upper bound: 96.5904930
time: 0.81 seconds

## BFS IS instance: IS_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -35.9391365, 60.1226463, -46.7894211, 78.8589554, -114.7980652, 106.9120483
1: -39.2804604, 52.0018272, -51.1519814, 67.0058212, -106.2862854, 103.1538086
2: -40.2224274, 51.8713303, -52.3858948, 67.1554642, -107.3778915, 104.2572250
3: -46.2532082, 60.3966446, -60.2768440, 77.6478806, -123.9010544, 120.6734848
4: -42.7411232, 60.1157494, -55.1636429, 77.8824539, -120.6235809, 115.2793884

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_B1_A1_A1_A1

### Relational analysis result of IS_B2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5499522, upper bound: 96.5807164
time: 0.84 seconds

## Relational analysis of IS_B2_B2_B1_A1_A1_A2

### Relational analysis result of IS_B2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5869614, upper bound: 96.5892486
time: 1.10 seconds

## BFS IS instance: IS_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -35.0173721, 58.5461388, -46.7894211, 78.8589554, -113.8763199, 105.3355408
1: -38.2671204, 50.4559860, -51.1519814, 67.0058212, -105.2729416, 101.6079636
2: -39.1966782, 50.3279610, -52.3858948, 67.1554642, -106.3521423, 102.7138519
3: -45.0551987, 58.5559807, -60.2768440, 77.6478806, -122.7030640, 118.8328171
4: -41.6632614, 58.3365135, -55.1636429, 77.8824539, -119.5457153, 113.5001526

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_B1_A1_A2_A1

### Relational analysis result of IS_B2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5499522, upper bound: 96.5807164
time: 0.93 seconds

## Relational analysis of IS_B2_B2_B1_A1_A2_A2

### Relational analysis result of IS_B2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5869614, upper bound: 96.5909660
time: 0.87 seconds

## BFS IS instance: IS_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -34.0779266, 57.4114685, -46.1612206, 78.0415955, -112.1195068, 103.5726776
1: -37.2440262, 49.3638725, -50.4673233, 66.2277527, -103.4717712, 99.8311768
2: -38.1722374, 49.2737236, -51.6925659, 66.3818436, -104.5540771, 100.9662933
3: -43.9307022, 57.2860870, -59.4975586, 76.7291641, -120.6598663, 116.7836304
4: -40.6252975, 57.0304909, -54.4438019, 76.9519577, -117.5772552, 111.4742889

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A2_A1_A1

### Relational analysis result of IS_B2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5077565, upper bound: 96.5766814
time: 0.90 seconds

## Relational analysis of IS_B2_B2_B1_A2_A1_A2

### Relational analysis result of IS_B2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5077565, upper bound: 96.5670420
time: 0.89 seconds

## BFS IS instance: IS_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -35.6753387, 59.9220886, -46.3039894, 78.3394241, -114.0147629, 106.2260437
1: -38.9712868, 51.4955025, -50.6369019, 66.5190125, -105.4902954, 102.1323929
2: -39.9152374, 51.4027214, -51.8644943, 66.6621857, -106.5774231, 103.2672043
3: -45.9741058, 59.7754860, -59.7156525, 77.0766830, -123.0507812, 119.4911346
4: -42.3825264, 59.5560837, -54.6583519, 77.2841797, -119.6667023, 114.2144318

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A2_A2_A1

### Relational analysis result of IS_B2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5107969, upper bound: 96.5766858
time: 1.06 seconds

## Relational analysis of IS_B2_B2_B1_A2_A2_A2

### Relational analysis result of IS_B2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5107969, upper bound: 96.5915732
time: 1.00 seconds

## BFS IS instance: IS_B2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -21.1572266, 41.2690582, -40.1067276, 68.4115601, -89.5687714, 81.3757858
1: -23.2374840, 33.9224701, -43.8479156, 57.7339668, -80.9714508, 77.7703857
2: -23.8651142, 33.5957642, -44.9056244, 57.7641373, -81.6292496, 78.5013885
3: -27.8600044, 39.2188759, -51.6687050, 66.8986969, -94.7586823, 90.8875809
4: -26.8484497, 38.3758736, -47.5603218, 66.8738327, -93.7222824, 85.9361801

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_B2_B2_B1_A1_B1

### Relational analysis result of IS_B2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5058367, upper bound: 96.5480309
time: 1.00 seconds

## Relational analysis of IS_B2_B2_B2_B1_A1_B2

### Relational analysis result of IS_B2_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5102032, upper bound: 96.5787353
time: 1.04 seconds

## BFS IS instance: IS_B2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -45.2934227, 76.3665237, -40.1067276, 68.4115601, -113.7049866, 116.4732513
1: -49.5006523, 64.7596054, -43.8479156, 57.7339668, -107.2346039, 108.6075211
2: -50.7261887, 64.8742065, -44.9056244, 57.7641373, -108.4903259, 109.7798309
3: -58.3333130, 75.0735626, -51.6687050, 66.8986969, -125.2319870, 126.7422638
4: -53.5064850, 75.2585373, -47.5603218, 66.8738327, -120.3803177, 122.8188629

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B2_B1_A2_A1

### Relational analysis result of IS_B2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5073900, upper bound: 96.5981170
time: 1.33 seconds

## Relational analysis of IS_B2_B2_B2_B1_A2_A2

### Relational analysis result of IS_B2_B2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5073900, upper bound: 96.5110285
time: 0.95 seconds

## BFS IS instance: IS_B2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -34.0544167, 57.3635292, -43.5565796, 74.4665298, -108.5209427, 100.9200974
1: -37.2184334, 49.3166656, -47.6642914, 62.8809738, -100.0994110, 96.9809494
2: -38.1481972, 49.2265663, -48.8466415, 63.0090294, -101.1572266, 98.0732040
3: -43.9024658, 57.2319260, -56.3230286, 72.8328857, -116.7353287, 113.5549393
4: -40.5974770, 56.9866066, -51.5112076, 72.9934998, -113.5909729, 108.4978027

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5078615, upper bound: 96.5769281
time: 1.05 seconds

## Relational analysis of IS_B2_B2_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5078615, upper bound: 96.5724941
time: 1.12 seconds

## BFS IS instance: IS_B2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -35.6210861, 59.8295822, -43.5505028, 74.5775299, -110.1986084, 103.3800583
1: -38.9113808, 51.4106407, -47.6673012, 62.9896164, -101.9009857, 99.0779419
2: -39.8571815, 51.3178864, -48.8545380, 63.1111069, -102.9682770, 100.1724243
3: -45.9043465, 59.6776505, -56.3538895, 72.9655685, -118.8698959, 116.0315247
4: -42.3209991, 59.4652939, -51.5552368, 73.1105270, -115.4315262, 111.0205231

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5109209, upper bound: 96.5797609
time: 1.03 seconds

## Relational analysis of IS_B2_B2_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5109209, upper bound: 96.5982053
time: 0.99 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.47 seconds
IS_B2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5498342, upper bound: 96.5782974
IS_B2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5498342, upper bound: 96.5782974
IS_B2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5868443, upper bound: 96.5868441
IS_B2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5868443, upper bound: 96.5885607
IS_B2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5470265, upper bound: 96.5765912
IS_B2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5890115, upper bound: 96.5863209
IS_B2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5476609, upper bound: 96.5798941
IS_B2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5896371, upper bound: 96.5896370
IS_B2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5104767, upper bound: 96.5759977
IS_B2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5104767, upper bound: 96.5768431
IS_B2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5104767, upper bound: 96.5885629
IS_B2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5104767, upper bound: 96.5906010
IS_B2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5576800, upper bound: 96.5788116
IS_B2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5967071, upper bound: 96.5871933
IS_B2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5583142, upper bound: 96.5820992
IS_B2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5973327, upper bound: 96.5904930
IS_B2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5499522, upper bound: 96.5807164
IS_B2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5869614, upper bound: 96.5892486
IS_B2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5499522, upper bound: 96.5807164
IS_B2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5869614, upper bound: 96.5909660
IS_B2_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5077565, upper bound: 96.5766814
IS_B2_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5077565, upper bound: 96.5670420
IS_B2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5107969, upper bound: 96.5766858
IS_B2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5107969, upper bound: 96.5915732
IS_B2_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5058367, upper bound: 96.5480309
IS_B2_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5102032, upper bound: 96.5787353
IS_B2_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5073900, upper bound: 96.5981170
IS_B2_B2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5073900, upper bound: 96.5110285
IS_B2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5078615, upper bound: 96.5769281
IS_B2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5078615, upper bound: 96.5724941
IS_B2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5109209, upper bound: 96.5797609
IS_B2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.47
Output dim: 4, lower bound: -96.5109209, upper bound: 96.5982053

## BFS IS instance: IS_B2_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -33.4145088, 56.9669456, -45.6192589, 77.5343170, -110.9488068, 102.5861969
1: -36.5507774, 49.0025673, -49.8882027, 65.6745453, -102.2253189, 98.8907394
2: -37.4616356, 48.8737946, -51.1045876, 65.8133011, -103.2749329, 99.9783783
3: -43.1748581, 56.8205643, -58.8566971, 76.0790482, -119.2538834, 115.6772537
4: -39.9546700, 56.5314217, -53.8708038, 76.2695541, -116.2242203, 110.4022217

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_B2_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5444110, upper bound: 96.5543364
time: 0.95 seconds

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_B2_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5444110, upper bound: 96.5543364
time: 1.04 seconds

## BFS IS instance: IS_B2_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -32.6562538, 55.5736923, -45.6192589, 77.5343170, -110.1905518, 101.1929474
1: -35.7144165, 47.6360664, -49.8882027, 65.6745453, -101.3889542, 97.5242615
2: -36.6125069, 47.5131683, -51.1045876, 65.8133011, -102.4258118, 98.6177521
3: -42.1799240, 55.2628708, -58.8566971, 76.0790482, -118.2589645, 114.1195602
4: -39.0570297, 54.9702377, -53.8708038, 76.2695541, -115.3265839, 108.8410416

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_B2_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5444110, upper bound: 96.5543364
time: 0.78 seconds

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_B2_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5444110, upper bound: 96.5543364
time: 0.98 seconds

## BFS IS instance: IS_B2_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -35.1985550, 59.7633057, -45.7611084, 77.8297119, -113.0282669, 105.5244141
1: -38.4966316, 51.4193840, -50.0567780, 65.9632339, -104.4598541, 101.4761658
2: -39.4110832, 51.2794838, -51.2750854, 66.0913315, -105.5024109, 102.5545654
3: -45.4831123, 59.6171722, -59.0732384, 76.4241943, -121.9072952, 118.6904144
4: -41.9378090, 59.3904457, -54.0848274, 76.5993881, -118.5371857, 113.4752426

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A1_B1_A2_A1_B1

### Relational analysis result of IS_B2_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5813842, upper bound: 96.5629166
time: 1.21 seconds

## Relational analysis of IS_B2_B1_A1_B1_A2_A1_B2

### Relational analysis result of IS_B2_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5813842, upper bound: 96.5868441
time: 0.97 seconds

## BFS IS instance: IS_B2_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -33.9301682, 57.7230453, -45.7611084, 77.8297119, -111.7598724, 103.4841537
1: -37.1016922, 49.4505501, -50.0567780, 65.9632339, -103.0649109, 99.5073242
2: -38.0086784, 49.3154602, -51.2750854, 66.0913315, -104.0999985, 100.5905457
3: -43.8558807, 57.3827972, -59.0732384, 76.4241943, -120.2800751, 116.4560394
4: -40.4838943, 57.1043358, -54.0848274, 76.5993881, -117.0832748, 111.1891556

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A1_B1_A2_A2_B1

### Relational analysis result of IS_B2_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5813842, upper bound: 96.5646386
time: 1.17 seconds

## Relational analysis of IS_B2_B1_A1_B1_A2_A2_B2

### Relational analysis result of IS_B2_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5813842, upper bound: 96.5885607
time: 0.85 seconds

## BFS IS instance: IS_B2_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -31.8284550, 54.4030762, -38.9815140, 67.1587448, -98.9871979, 93.3845825
1: -34.8120804, 46.6103439, -42.6392021, 56.4694366, -91.2815170, 89.2495422
2: -35.7000046, 46.4764748, -43.6747169, 56.4825821, -92.1825867, 90.1511841
3: -41.1457825, 54.0604630, -50.3124161, 65.4065018, -106.5522766, 104.3728714
4: -38.1357651, 53.7567978, -46.3209152, 65.3313217, -103.4670792, 100.0777130

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_B1

### Relational analysis result of IS_B2_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5459536, upper bound: 96.5714624
time: 0.99 seconds

## Relational analysis of IS_B2_B1_A1_B2_B1_A1_B2

### Relational analysis result of IS_B2_B1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5470034, upper bound: 96.5740675
time: 0.95 seconds

## BFS IS instance: IS_B2_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -33.0796814, 56.3383713, -39.0678711, 67.3653030, -100.4449844, 95.4062424
1: -36.1632919, 48.2822456, -42.7381554, 56.6670952, -92.8303604, 91.0204010
2: -37.0564194, 48.1336594, -43.7819748, 56.6753731, -93.7317657, 91.9156342
3: -42.7572556, 56.0186348, -50.4478722, 65.6420975, -108.3993530, 106.4664993
4: -39.5133781, 55.7389107, -46.4656830, 65.5628357, -105.0762177, 102.2045898

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5856954, upper bound: 96.5856953
time: 1.10 seconds

## Relational analysis of IS_B2_B1_A1_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5856954, upper bound: 96.5863209
time: 1.03 seconds

## BFS IS instance: IS_B2_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -33.4319534, 56.7833214, -42.9490814, 73.8768539, -107.3088074, 99.7324066
1: -36.5543785, 48.6932831, -47.0158806, 62.2497177, -98.8040924, 95.7091675
2: -37.4765091, 48.5840874, -48.1866684, 62.3603058, -99.8367920, 96.7707367
3: -43.1722488, 56.4961967, -55.6064110, 72.0858459, -115.2580719, 112.1026077
4: -39.9387970, 56.2211075, -50.8620529, 72.2152176, -112.1540070, 107.0831451

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_A1

### Relational analysis result of IS_B2_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5466644, upper bound: 96.5611174
time: 1.28 seconds

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_B1

### Relational analysis result of IS_B2_B1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5438684, upper bound: 96.5668906
time: 1.03 seconds

## Relational analysis of IS_B2_B1_A1_B2_B2_A1_B2

### Relational analysis result of IS_B2_B1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5476503, upper bound: 96.5788543
time: 0.89 seconds

## BFS IS instance: IS_B2_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -34.8316994, 58.8424149, -42.9455795, 73.9901047, -108.8218002, 101.7879944
1: -38.0594101, 50.4835587, -47.0218086, 62.3612938, -100.4206696, 97.5053635
2: -38.9898720, 50.3663483, -48.1973648, 62.4646530, -101.4545212, 98.5636826
3: -44.9390678, 58.5930328, -55.6408806, 72.2217712, -117.1608429, 114.2339020
4: -41.4638977, 58.3519630, -50.9091377, 72.3352509, -113.7991486, 109.2611008

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B2_B2_A2_A1

### Relational analysis result of IS_B2_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5856954, upper bound: 96.5890114
time: 0.92 seconds

## Relational analysis of IS_B2_B1_A1_B2_B2_A2_A2

### Relational analysis result of IS_B2_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5856954, upper bound: 96.5896370
time: 1.02 seconds

## BFS IS instance: IS_B2_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -22.5569458, 43.1898117, -46.2494240, 78.3527451, -100.9096909, 89.4392166
1: -24.7445717, 35.7115326, -50.5749245, 66.4542007, -91.1987457, 86.2864380
2: -25.3966751, 35.4250870, -51.7997055, 66.5883789, -91.9850540, 87.2247925
3: -29.5906868, 41.2766342, -59.6375389, 76.9996033, -106.5902786, 100.9141464
4: -28.3221054, 40.4803848, -54.5916595, 77.2020187, -105.5241241, 95.0720444

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A2_B1_A1_A1_A1

### Relational analysis result of IS_B2_B1_A2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4517411, upper bound: 96.4926401
time: 1.13 seconds

## Relational analysis of IS_B2_B1_A2_B1_A1_A1_A2

### Relational analysis result of IS_B2_B1_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5137415, upper bound: 96.5732786
time: 1.48 seconds

## BFS IS instance: IS_B2_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -22.0632725, 42.4937210, -46.2494240, 78.3527451, -100.4160156, 88.7431488
1: -24.2183628, 35.0233841, -50.5749245, 66.4542007, -90.6725464, 85.5982971
2: -24.8598690, 34.6991234, -51.7997055, 66.5883789, -91.4482498, 86.4988251
3: -28.9810963, 40.5176315, -59.6375389, 76.9996033, -105.9806976, 100.1551666
4: -27.8329735, 39.6638184, -54.5916595, 77.2020187, -105.0349884, 94.2554779

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_B2_B1_A2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4517411, upper bound: 96.4926401
time: 1.12 seconds

## Relational analysis of IS_B2_B1_A2_B1_A1_A2_A2

### Relational analysis result of IS_B2_B1_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5137415, upper bound: 96.5743689
time: 1.00 seconds

## BFS IS instance: IS_B2_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -46.7894211, 78.8589554, -46.2494240, 78.3527451, -125.1421509, 125.1083679
1: -51.1519814, 67.0058212, -50.5749245, 66.4542007, -117.6061859, 117.5807495
2: -52.3858948, 67.1554642, -51.7997055, 66.5883789, -118.9742737, 118.9551697
3: -60.2768440, 77.6478806, -59.6375389, 76.9996033, -137.2764435, 137.2854156
4: -55.1636429, 77.8824539, -54.5916595, 77.2020187, -132.3656616, 132.4741211

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A2_B1_A2_A1_A1

### Relational analysis result of IS_B2_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5575007, upper bound: 96.5576448
time: 1.03 seconds

## Relational analysis of IS_B2_B1_A2_B1_A2_A1_A2

### Relational analysis result of IS_B2_B1_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5885605, upper bound: 96.5868388
time: 1.09 seconds

## BFS IS instance: IS_B2_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -46.0311928, 77.7005310, -46.2494240, 78.3527451, -124.3839340, 123.9499359
1: -50.3242874, 65.8815689, -50.5749245, 66.4542007, -116.7784805, 116.4564972
2: -51.5641251, 66.0073242, -51.7997055, 66.5883789, -118.1525040, 117.8070297
3: -59.3308029, 76.3739243, -59.6375389, 76.9996033, -136.3304138, 136.0114441
4: -54.3605728, 76.5541840, -54.5916595, 77.2020187, -131.5625916, 131.1458130

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_B2_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5838046, upper bound: 96.5652937
time: 0.98 seconds

## Relational analysis of IS_B2_B1_A2_B1_A2_A2_B2

### Relational analysis result of IS_B2_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5885605, upper bound: 96.5868388
time: 0.74 seconds

## BFS IS instance: IS_B2_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -32.3685684, 54.9189644, -38.9815140, 67.1587448, -99.5273132, 93.9004669
1: -35.3897209, 47.1713104, -42.6392021, 56.4694366, -91.8591614, 89.8104935
2: -36.2855682, 47.0525970, -43.6747169, 56.4825821, -92.7681503, 90.7273102
3: -41.7866745, 54.7226257, -50.3124161, 65.4065018, -107.1931763, 105.0350342
4: -38.7164268, 54.4409981, -46.3209152, 65.3313217, -104.0477448, 100.7619171

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_B1_A2_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5575210, upper bound: 96.5709861
time: 1.03 seconds

## Relational analysis of IS_B2_B1_A2_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A2_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5444773, upper bound: 96.5762305
time: 0.90 seconds

## Relational analysis of IS_B2_B1_A2_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5561275, upper bound: 96.5783372
time: 0.75 seconds

## BFS IS instance: IS_B2_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -33.6605072, 56.8821068, -39.0678711, 67.3653030, -101.0258102, 95.9499817
1: -36.7830086, 48.8665581, -42.7381554, 56.6670952, -93.4500809, 91.6047134
2: -37.6837234, 48.7363853, -43.7819748, 56.6753731, -94.3590927, 92.5183563
3: -43.4367027, 56.7067642, -50.4478722, 65.6420975, -109.0787964, 107.1546249
4: -40.1267014, 56.4547462, -46.4656830, 65.5628357, -105.6895370, 102.9204178

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A2_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5963886, upper bound: 96.5867011
time: 1.55 seconds

## Relational analysis of IS_B2_B1_A2_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5963886, upper bound: 96.5871933
time: 1.30 seconds

## BFS IS instance: IS_B2_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -34.0139542, 57.3229485, -42.9490814, 73.8768539, -107.8908081, 100.2720032
1: -37.1753082, 49.2775154, -47.0158806, 62.2497177, -99.4250259, 96.2933960
2: -38.1049232, 49.1857338, -48.1866684, 62.3603058, -100.4652252, 97.3723907
3: -43.8555031, 57.1861343, -55.6064110, 72.0858459, -115.9413452, 112.7925415
4: -40.5558510, 56.9375992, -50.8620529, 72.2152176, -112.7710648, 107.7996368

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_B1_A2_B2_B2_A1_B1

### Relational analysis result of IS_B2_B1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5545204, upper bound: 96.5691433
time: 1.15 seconds

## Relational analysis of IS_B2_B1_A2_B2_B2_A1_B2

### Relational analysis result of IS_B2_B1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5583037, upper bound: 96.5811011
time: 1.02 seconds

## BFS IS instance: IS_B2_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -35.5787468, 59.7863197, -42.9455795, 73.9901047, -109.5688477, 102.7318802
1: -38.8662071, 51.3690834, -47.0218086, 62.3612938, -101.2274780, 98.3908691
2: -39.8118210, 51.2746582, -48.1973648, 62.4646530, -102.2764740, 99.4720230
3: -45.8552666, 59.6290016, -55.6408806, 72.2217712, -118.0770416, 115.2698822
4: -42.2772675, 59.4135704, -50.9091377, 72.3352509, -114.6125183, 110.3227081

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A2_B2_B2_A2_A1

### Relational analysis result of IS_B2_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5963886, upper bound: 96.5900177
time: 0.98 seconds

## Relational analysis of IS_B2_B1_A2_B2_B2_A2_A2

### Relational analysis result of IS_B2_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5963886, upper bound: 96.5905093
time: 1.09 seconds

## BFS IS instance: IS_B2_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -33.4145088, 56.9669456, -46.1612206, 78.0415955, -111.4560852, 103.1281586
1: -36.5507774, 49.0025673, -50.4673233, 66.2277527, -102.7785187, 99.4698639
2: -37.4616356, 48.8737946, -51.6925659, 66.3818436, -103.8434753, 100.5663605
3: -43.1748581, 56.8205643, -59.4975586, 76.7291641, -119.9040070, 116.3181152
4: -39.9546700, 56.5314217, -54.4438019, 76.9519577, -116.9066238, 110.9752197

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_B2_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4608870, upper bound: 96.5680880
time: 1.07 seconds

## Relational analysis of IS_B2_B2_B1_A1_A1_A1_A2

### Relational analysis result of IS_B2_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4608870, upper bound: 96.5774132
time: 0.94 seconds

## BFS IS instance: IS_B2_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -35.1985550, 59.7633057, -46.3039894, 78.3394241, -113.5379791, 106.0672913
1: -38.4966316, 51.4193840, -50.6369019, 66.5190125, -105.0156403, 102.0562897
2: -39.4110832, 51.2794838, -51.8644943, 66.6621857, -106.0732727, 103.1439667
3: -45.4831123, 59.6171722, -59.7156525, 77.0766830, -122.5597916, 119.3328247
4: -41.9378090, 59.3904457, -54.6583519, 77.2841797, -119.2219849, 114.0487671

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_B2_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5825004, upper bound: 96.5684599
time: 1.03 seconds

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_B2

### Relational analysis result of IS_B2_B2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5825004, upper bound: 96.5892486
time: 1.17 seconds

## BFS IS instance: IS_B2_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -32.6562538, 55.5736923, -46.1612206, 78.0415955, -110.6978455, 101.7349091
1: -35.7144165, 47.6360664, -50.4673233, 66.2277527, -101.9421387, 98.1033936
2: -36.6125069, 47.5131683, -51.6925659, 66.3818436, -102.9943542, 99.2057343
3: -42.1799240, 55.2628708, -59.4975586, 76.7291641, -118.9090881, 114.7604065
4: -39.0570297, 54.9702377, -54.4438019, 76.9519577, -116.0089874, 109.4140396

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_B2_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5455223, upper bound: 96.5599158
time: 0.94 seconds

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_B2

### Relational analysis result of IS_B2_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5455223, upper bound: 96.5807164
time: 0.92 seconds

## BFS IS instance: IS_B2_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -33.7916336, 57.3896255, -46.3039894, 78.3394241, -112.1310577, 103.6935883
1: -36.9463844, 49.2017250, -50.6369019, 66.5190125, -103.4653931, 99.8386230
2: -37.8473511, 49.0609818, -51.8644943, 66.6621857, -104.5095367, 100.9254379
3: -43.6603508, 57.0959473, -59.7156525, 77.0766830, -120.7370300, 116.8115997
4: -40.3187370, 56.8183365, -54.6583519, 77.2841797, -117.6029205, 111.4766846

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_B2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_B2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_B2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_B1_A1_A2_A2_B1

### Relational analysis result of IS_B2_B2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5963585, upper bound: 96.5701820
time: 0.93 seconds

## Relational analysis of IS_B2_B2_B1_A1_A2_A2_B2

### Relational analysis result of IS_B2_B2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5963585, upper bound: 96.5909662
time: 0.97 seconds

## BFS IS instance: IS_B2_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -21.7965202, 42.1699715, -46.1612206, 78.0415955, -99.8381042, 88.3311920
1: -23.9183350, 34.7588806, -50.4673233, 66.2277527, -90.1460800, 85.2262039
2: -24.5837307, 34.4497070, -51.6925659, 66.3818436, -90.9655685, 86.1422729
3: -28.6563644, 40.2012863, -59.4975586, 76.7291641, -105.3855286, 99.6988373
4: -27.6126938, 39.3420334, -54.4438019, 76.9519577, -104.5646515, 93.7858353

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_B2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_B2_B1_A2_A1_A1_A1

### Relational analysis result of IS_B2_B2_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5012040, upper bound: 96.5749556
time: 0.76 seconds

## Relational analysis of IS_B2_B2_B1_A2_A1_A1_A2

### Relational analysis result of IS_B2_B2_B1_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5077565, upper bound: 96.5069168
time: 1.12 seconds

## BFS IS instance: IS_B2_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -44.7805901, 76.0971909, -46.1612206, 78.0415955, -122.8221893, 122.2584076
1: -48.9469833, 64.2719727, -50.4673233, 66.2277527, -115.1747284, 114.7392883
2: -50.1723518, 64.4224472, -51.6925659, 66.3818436, -116.5541992, 116.1150055
3: -57.7532501, 74.4294891, -59.4975586, 76.7291641, -134.4824066, 133.9269867
4: -52.8796692, 74.6202011, -54.4438019, 76.9519577, -129.8316345, 129.0639954

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A2_A1_A2_A1

### Relational analysis result of IS_B2_B2_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5077042, upper bound: 96.5670420
time: 0.88 seconds

## Relational analysis of IS_B2_B2_B1_A2_A1_A2_A2

### Relational analysis result of IS_B2_B2_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5077042, upper bound: 96.5670429
time: 0.95 seconds

## BFS IS instance: IS_B2_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -23.0129623, 44.1977081, -46.3039894, 78.3394241, -101.3523865, 90.5016937
1: -25.2573071, 36.4886971, -50.6369019, 66.5190125, -91.7763214, 87.1255951
2: -25.9593773, 36.1812096, -51.8644943, 66.6621857, -92.6215668, 88.0456848
3: -30.3101387, 42.1957436, -59.7156525, 77.0766830, -107.3868256, 101.9113770
4: -28.9936409, 41.4195404, -54.6583519, 77.2841797, -106.2778168, 96.0778961

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A2_A2_A1_A1

### Relational analysis result of IS_B2_B2_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5067418, upper bound: 96.5765064
time: 0.92 seconds

## Relational analysis of IS_B2_B2_B1_A2_A2_A1_A2

### Relational analysis result of IS_B2_B2_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5067418, upper bound: 96.5765067
time: 1.10 seconds

## BFS IS instance: IS_B2_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -46.0060120, 78.6338425, -46.3039894, 78.3394241, -124.3454361, 124.9378357
1: -50.3193398, 66.4030457, -50.6369019, 66.5190125, -116.8383484, 117.0399475
2: -51.5729446, 66.5129471, -51.8644943, 66.6621857, -118.2351303, 118.3774338
3: -59.4934883, 76.8816910, -59.7156525, 77.0766830, -136.5701752, 136.5973511
4: -54.4370003, 77.0852509, -54.6583519, 77.2841797, -131.7211761, 131.7436066

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A2_A2_A2_A1

### Relational analysis result of IS_B2_B2_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5067418, upper bound: 96.5893659
time: 1.00 seconds

## Relational analysis of IS_B2_B2_B1_A2_A2_A2_A2

### Relational analysis result of IS_B2_B2_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5067418, upper bound: 96.5915732
time: 0.98 seconds

## BFS IS instance: IS_B2_B2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -19.9496384, 39.8337288, -34.4946251, 60.7862587, -80.7359009, 74.3283463
1: -21.9530640, 32.5810394, -37.7921638, 51.0146599, -72.9677277, 70.3731995
2: -22.5519867, 32.2451172, -38.7400398, 50.9881783, -73.5401611, 70.9851532
3: -26.4104347, 37.6590424, -44.7622147, 59.0615959, -85.4720306, 82.4212494
4: -25.6546669, 36.7850723, -41.4250832, 58.8575974, -84.5122528, 78.2101593

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A1_B1_A1

### Relational analysis result of IS_B2_B2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5117522, upper bound: 96.5467917
time: 0.85 seconds

## Relational analysis of IS_B2_B2_B2_B1_A1_B1_A2

### Relational analysis result of IS_B2_B2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5079995, upper bound: 96.5450868
time: 0.86 seconds

## BFS IS instance: IS_B2_B2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -21.0413094, 41.1229248, -39.4515686, 67.5562286, -88.5975342, 80.5744934
1: -23.1137695, 33.7879143, -43.1380615, 56.9545212, -80.0682907, 76.9259796
2: -23.7386646, 33.4612389, -44.1874390, 56.9779816, -80.7166443, 77.6486664
3: -27.7199383, 39.0606346, -50.8565369, 65.9853592, -93.7052917, 89.9171753
4: -26.7266159, 38.2165527, -46.8466034, 65.9359207, -92.6625214, 85.0631561

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B2_B2_B1_A1_B2_A1

### Relational analysis result of IS_B2_B2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4956337, upper bound: 96.5639426
time: 1.23 seconds

## Relational analysis of IS_B2_B2_B2_B1_A1_B2_A2

### Relational analysis result of IS_B2_B2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4956337, upper bound: 96.5639426
time: 1.17 seconds

## BFS IS instance: IS_B2_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -41.2451363, 69.9344482, -40.1067276, 68.4115601, -109.6566925, 110.0411758
1: -45.0598869, 59.0772896, -43.8479156, 57.7339668, -102.7938538, 102.9252014
2: -46.1511917, 59.1296959, -44.9056244, 57.7641373, -103.9153214, 104.0353241
3: -53.0441589, 68.4498901, -51.6687050, 66.8986969, -119.9428253, 120.1185913
4: -48.7965202, 68.4947357, -47.5603218, 66.8738327, -115.6703491, 116.0550461

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_B2_B1_A2_A1_A1

### Relational analysis result of IS_B2_B2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5854729, upper bound: 96.5970951
time: 0.99 seconds

## Relational analysis of IS_B2_B2_B2_B1_A2_A1_A2

### Relational analysis result of IS_B2_B2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5854729, upper bound: 96.5979526
time: 1.03 seconds

## BFS IS instance: IS_B2_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -21.6903305, 42.0207443, -43.5565796, 74.4665298, -96.1568604, 85.5773163
1: -23.8018875, 34.6166611, -47.6642914, 62.8809738, -86.6828461, 82.2809525
2: -24.4680443, 34.3044281, -48.8466415, 63.0090294, -87.4770737, 83.1510620
3: -28.5209675, 40.0370674, -56.3230286, 72.8328857, -101.3538513, 96.3600922
4: -27.4944305, 39.1766777, -51.5112076, 72.9934998, -100.4879227, 90.6878738

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_B2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_A1_A1_A1

### Relational analysis result of IS_B2_B2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5046141, upper bound: 96.5752178
time: 1.11 seconds

## Relational analysis of IS_B2_B2_B2_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_B2_B2_B2_A1_A1_A1

### Relational analysis result of IS_B2_B2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5013130, upper bound: 96.5758442
time: 1.40 seconds

## Relational analysis of IS_B2_B2_B2_B2_A1_A1_A2

### Relational analysis result of IS_B2_B2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5078615, upper bound: 96.5768431
time: 1.22 seconds

## BFS IS instance: IS_B2_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -44.5028839, 75.5278244, -43.5565796, 74.4665298, -118.9694138, 119.0844040
1: -48.6407585, 63.8087540, -47.6642914, 62.8809738, -111.5217285, 111.4730453
2: -49.8569565, 63.9516678, -48.8466415, 63.0090294, -112.8659821, 112.7983017
3: -57.3827667, 73.9095306, -56.3230286, 72.8328857, -130.2156525, 130.2325439
4: -52.5661316, 74.0998383, -51.5112076, 72.9934998, -125.5596237, 125.6110306

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B2_B2_A1_A2_A1

### Relational analysis result of IS_B2_B2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5076215, upper bound: 96.5724941
time: 1.06 seconds

## Relational analysis of IS_B2_B2_B2_B2_A1_A2_A2

### Relational analysis result of IS_B2_B2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5076215, upper bound: 96.5654334
time: 1.19 seconds

## BFS IS instance: IS_B2_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -22.8709564, 44.0069427, -43.5505028, 74.5775299, -97.4484863, 87.5574493
1: -25.1049690, 36.3109093, -47.6673012, 62.9896164, -88.0945892, 83.9782104
2: -25.8061676, 35.9970322, -48.8545380, 63.1111069, -88.9172668, 84.8515701
3: -30.1369858, 41.9892426, -56.3538895, 72.9655685, -103.1025543, 98.3431244
4: -28.8420792, 41.2137756, -51.5552368, 73.1105270, -101.9526062, 92.7689819

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_B2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B2_B2_A2_A1_A1

### Relational analysis result of IS_B2_B2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5048798, upper bound: 96.5752045
time: 1.21 seconds

## Relational analysis of IS_B2_B2_B2_B2_A2_A1_A2

### Relational analysis result of IS_B2_B2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5048798, upper bound: 96.5797609
time: 0.86 seconds

## BFS IS instance: IS_B2_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -45.7723999, 78.3666000, -43.5505028, 74.5775299, -120.3499298, 121.9170990
1: -50.0722198, 66.1604233, -47.6673012, 62.9896164, -113.0618210, 113.8277130
2: -51.3261719, 66.2619019, -48.8545380, 63.1111069, -114.4372482, 115.1164398
3: -59.2260361, 76.6043472, -56.3538895, 72.9655685, -132.1916046, 132.9582214
4: -54.1989784, 76.7956924, -51.5552368, 73.1105270, -127.3095093, 128.3509064

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B2_B2_A2_A2_A1

### Relational analysis result of IS_B2_B2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5066592, upper bound: 96.5934975
time: 0.98 seconds

## Relational analysis of IS_B2_B2_B2_B2_A2_A2_A2

### Relational analysis result of IS_B2_B2_B2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5066592, upper bound: 96.5104409
time: 0.85 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.33 seconds
IS_B2_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5444110, upper bound: 96.5543364
IS_B2_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5444110, upper bound: 96.5543364
IS_B2_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5444110, upper bound: 96.5543364
IS_B2_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5444110, upper bound: 96.5543364
IS_B2_B1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5813842, upper bound: 96.5629166
IS_B2_B1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5813842, upper bound: 96.5868441
IS_B2_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5813842, upper bound: 96.5646386
IS_B2_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5813842, upper bound: 96.5885607
IS_B2_B1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5459536, upper bound: 96.5714624
IS_B2_B1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5470034, upper bound: 96.5740675
IS_B2_B1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5856954, upper bound: 96.5856953
IS_B2_B1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5856954, upper bound: 96.5863209
IS_B2_B1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5438684, upper bound: 96.5668906
IS_B2_B1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5476503, upper bound: 96.5788543
IS_B2_B1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5856954, upper bound: 96.5890114
IS_B2_B1_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5856954, upper bound: 96.5896370
IS_B2_B1_A2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.4517411, upper bound: 96.4926401
IS_B2_B1_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5137415, upper bound: 96.5732786
IS_B2_B1_A2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.4517411, upper bound: 96.4926401
IS_B2_B1_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5137415, upper bound: 96.5743689
IS_B2_B1_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5575007, upper bound: 96.5576448
IS_B2_B1_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5885605, upper bound: 96.5868388
IS_B2_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5838046, upper bound: 96.5652937
IS_B2_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5885605, upper bound: 96.5868388
IS_B2_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5444773, upper bound: 96.5762305
IS_B2_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5561275, upper bound: 96.5783372
IS_B2_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5963886, upper bound: 96.5867011
IS_B2_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5963886, upper bound: 96.5871933
IS_B2_B1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5545204, upper bound: 96.5691433
IS_B2_B1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5583037, upper bound: 96.5811011
IS_B2_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5963886, upper bound: 96.5900177
IS_B2_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5963886, upper bound: 96.5905093
IS_B2_B2_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.4608870, upper bound: 96.5680880
IS_B2_B2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.4608870, upper bound: 96.5774132
IS_B2_B2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5825004, upper bound: 96.5684599
IS_B2_B2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5825004, upper bound: 96.5892486
IS_B2_B2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5455223, upper bound: 96.5599158
IS_B2_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5455223, upper bound: 96.5807164
IS_B2_B2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5963585, upper bound: 96.5701820
IS_B2_B2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5963585, upper bound: 96.5909662
IS_B2_B2_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5012040, upper bound: 96.5749556
IS_B2_B2_B1_A2_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5077565, upper bound: 96.5069168
IS_B2_B2_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5077042, upper bound: 96.5670420
IS_B2_B2_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5077042, upper bound: 96.5670429
IS_B2_B2_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5067418, upper bound: 96.5765064
IS_B2_B2_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5067418, upper bound: 96.5765067
IS_B2_B2_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5067418, upper bound: 96.5893659
IS_B2_B2_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5067418, upper bound: 96.5915732
IS_B2_B2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5117522, upper bound: 96.5467917
IS_B2_B2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5079995, upper bound: 96.5450868
IS_B2_B2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.4956337, upper bound: 96.5639426
IS_B2_B2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.4956337, upper bound: 96.5639426
IS_B2_B2_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5854729, upper bound: 96.5970951
IS_B2_B2_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5854729, upper bound: 96.5979526
IS_B2_B2_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5013130, upper bound: 96.5758442
IS_B2_B2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5078615, upper bound: 96.5768431
IS_B2_B2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5076215, upper bound: 96.5724941
IS_B2_B2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5076215, upper bound: 96.5654334
IS_B2_B2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5048798, upper bound: 96.5752045
IS_B2_B2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5048798, upper bound: 96.5797609
IS_B2_B2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5066592, upper bound: 96.5934975
IS_B2_B2_B2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.33
Output dim: 4, lower bound: -96.5066592, upper bound: 96.5104409

## BFS IS instance: IS_B2_B1_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -33.4145088, 56.9669456, -43.8958893, 75.2677841, -108.6822815, 100.8628159
1: -36.5507774, 49.0025673, -48.0123100, 63.5206261, -100.0714035, 97.0148544
2: -37.4616356, 48.8737946, -49.2027130, 63.6686783, -101.1303101, 98.0765076
3: -43.1748581, 56.8205643, -56.7138481, 73.5384140, -116.7132645, 113.5344086
4: -39.9546700, 56.5314217, -51.9174423, 73.6867752, -113.6414413, 108.4488678

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_B2_B1_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5563386, upper bound: 96.5540513
time: 0.92 seconds

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_B2_B1_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5530001, upper bound: 96.5529999
time: 0.96 seconds

## BFS IS instance: IS_B2_B1_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -33.4145088, 56.9669456, -45.3726845, 78.1260071, -111.5404968, 102.3396225
1: -36.5507774, 49.0025673, -49.6780663, 65.9792175, -102.5299988, 98.6806107
2: -37.4616356, 48.8737946, -50.8755875, 66.0840607, -103.5456924, 99.7493744
3: -43.1748581, 56.8205643, -58.8053551, 76.3731079, -119.5479507, 115.6259079
4: -39.9546700, 56.5314217, -53.7611275, 76.5278320, -116.4824829, 110.2925491

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B2_A1

### Relational analysis result of IS_B2_B1_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5380663, upper bound: 96.5776740
time: 1.12 seconds

## Relational analysis of IS_B2_B1_A1_B1_A1_A1_B2_A2

### Relational analysis result of IS_B2_B1_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5548145, upper bound: 96.5792157
time: 1.44 seconds

## BFS IS instance: IS_B2_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -32.6562538, 55.5736923, -43.8958893, 75.2677841, -107.9240265, 99.4695740
1: -35.7144165, 47.6360664, -48.0123100, 63.5206261, -99.2350388, 95.6483765
2: -36.6125069, 47.5131683, -49.2027130, 63.6686783, -100.2811890, 96.7158813
3: -42.1799240, 55.2628708, -56.7138481, 73.5384140, -115.7183380, 111.9767151
4: -39.0570297, 54.9702377, -51.9174423, 73.6867752, -112.7438049, 106.8876801

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_B2_B1_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5443662, upper bound: 96.5542943
time: 0.85 seconds

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_B2_B1_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5433146, upper bound: 96.5509611
time: 0.85 seconds

## BFS IS instance: IS_B2_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -32.6562538, 55.5736923, -45.3726845, 78.1260071, -110.7822495, 100.9463806
1: -35.7144165, 47.6360664, -49.6780663, 65.9792175, -101.6936340, 97.3141327
2: -36.6125069, 47.5131683, -50.8755875, 66.0840607, -102.6965561, 98.3887558
3: -42.1799240, 55.2628708, -58.8053551, 76.3731079, -118.5530319, 114.0682068
4: -39.0570297, 54.9702377, -53.7611275, 76.5278320, -115.5848618, 108.7313690

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=108.20734405517578
rel_dist={4: [-96.61854736505002, 96.61854736505003]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6137199, upper bound: 96.6021268
time: 1.02 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6179584, upper bound: 96.6179584
time: 1.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.30 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 2.30
Output dim: 4, lower bound: -96.6137199, upper bound: 96.6021268
IS_B2, status: Status.UNKNOWN, split count: 1, time: 2.30
Output dim: 4, lower bound: -96.6179584, upper bound: 96.6179584

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -37.0113792, 61.2733002, -35.6634521, 59.5843353, -96.5957184, 96.9367523
1: -40.4089966, 52.9375572, -38.9564934, 51.3453674, -91.7543640, 91.8940353
2: -41.3972626, 52.8423309, -39.9247971, 51.2532921, -92.6505585, 92.7671280
3: -47.5014229, 61.5331345, -45.8759232, 59.6697502, -107.1711731, 107.4090576
4: -43.9357872, 61.2771530, -42.4524384, 59.3787651, -103.3145370, 103.7295837

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5146912, upper bound: 96.5474428
time: 0.99 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6053322, upper bound: 96.5759321
time: 0.73 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -37.2466812, 61.5535889, -37.2158546, 62.0344048, -99.2810593, 98.7694244
1: -40.6567612, 53.2141724, -40.6336594, 53.4205246, -94.0772858, 93.8478241
2: -41.6503181, 53.1111488, -41.6190872, 53.3239632, -94.9742813, 94.7302322
3: -47.7727623, 61.8547058, -47.8585129, 62.0749512, -109.8477173, 109.7132034
4: -44.1899109, 61.6042290, -44.1665001, 61.8375549, -106.0274658, 105.7707291

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6102527, upper bound: 96.6170821
time: 1.00 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6177553, upper bound: 96.6177562
time: 0.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.20 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 4, lower bound: -96.5146912, upper bound: 96.5474428
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 4, lower bound: -96.6053322, upper bound: 96.5759321
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 4, lower bound: -96.6102527, upper bound: 96.6170821
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 4, lower bound: -96.6177553, upper bound: 96.6177562

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -22.8797913, 43.6451569, -26.5716782, 47.6872330, -70.5670013, 70.2168274
1: -25.0808849, 36.0795708, -29.0443325, 39.8807411, -64.9616241, 65.1239014
2: -25.7734623, 35.7657967, -29.7919044, 39.6765289, -65.4499893, 65.5576935
3: -29.9764824, 41.7457657, -34.3938103, 46.2388496, -76.2153244, 76.1395721
4: -28.7990532, 40.9032440, -32.2585220, 45.5413361, -74.3403854, 73.1617661

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5131282, upper bound: 96.5408059
time: 0.97 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5144530, upper bound: 96.5472222
time: 0.88 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -46.6079865, 78.4932327, -34.0134354, 57.2870483, -103.8950348, 112.5066681
1: -50.9164276, 66.4974518, -37.1634903, 49.2232819, -100.1397095, 103.6609344
2: -52.1821976, 66.6640396, -38.0933533, 49.1455421, -101.3277435, 104.7573929
3: -59.9979401, 77.0394440, -43.8434563, 57.0825615, -117.0804977, 120.8829041
4: -54.9254036, 77.3079529, -40.5155106, 56.8905067, -111.8159027, 117.8234634

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5981235, upper bound: 96.5749532
time: 0.76 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6051031, upper bound: 96.5757024
time: 0.87 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -36.6144981, 60.9541779, -37.1886063, 62.0036430, -98.6181335, 98.1427841
1: -39.9822845, 52.5779991, -40.6044235, 53.3916550, -93.3739243, 93.1824188
2: -40.9669838, 52.4535255, -41.5895500, 53.2944450, -94.2614288, 94.0430756
3: -47.0317841, 61.1034889, -47.8261566, 62.0409775, -109.0727539, 108.9296265
4: -43.5240402, 60.8214378, -44.1372528, 61.8020515, -105.3260803, 104.9586945

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6052107, upper bound: 96.6095911
time: 0.97 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6095911, upper bound: 96.6095911
time: 1.00 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -37.2058563, 61.5123062, -37.2158546, 62.0344048, -99.2402496, 98.7281494
1: -40.6132507, 53.1743889, -40.6336594, 53.4205246, -94.0337677, 93.8080292
2: -41.6065598, 53.0697670, -41.6190872, 53.3239632, -94.9305267, 94.6888580
3: -47.7253532, 61.8078613, -47.8585129, 62.0749512, -109.8003006, 109.6663513
4: -44.1478081, 61.5545120, -44.1665001, 61.8375549, -105.9853592, 105.7210083

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5187834, upper bound: 96.5597155
time: 1.09 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6144582, upper bound: 96.6144592
time: 0.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.12 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 4, lower bound: -96.5131282, upper bound: 96.5408059
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 4, lower bound: -96.5144530, upper bound: 96.5472222
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 4, lower bound: -96.5981235, upper bound: 96.5749532
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 4, lower bound: -96.6051031, upper bound: 96.5757024
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 4, lower bound: -96.6052107, upper bound: 96.6095911
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 4, lower bound: -96.6095911, upper bound: 96.6095911
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 4, lower bound: -96.5187834, upper bound: 96.5597155
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 4, lower bound: -96.6144582, upper bound: 96.6144592

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -22.8583374, 43.6230011, -26.0336666, 47.2540054, -70.1123428, 69.6566696
1: -25.0586834, 36.0585289, -28.4766407, 39.3937798, -64.4524612, 64.5351715
2: -25.7505493, 35.7442818, -29.2125053, 39.1688919, -64.9194412, 64.9567871
3: -29.9527874, 41.7208443, -33.7813759, 45.6544456, -75.6072159, 75.5021973
4: -28.7765503, 40.8779526, -31.6994438, 44.9448776, -73.7214279, 72.5773926

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5131282, upper bound: 96.5377879
time: 1.02 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5131282, upper bound: 96.5408059
time: 0.94 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -22.8797913, 43.6451569, -26.5381927, 47.6553535, -70.5351334, 70.1833496
1: -25.0808849, 36.0795708, -29.0092030, 39.8505592, -64.9314423, 65.0887756
2: -25.7734623, 35.7657967, -29.7562714, 39.6448326, -65.4182968, 65.5220413
3: -29.9764824, 41.7457657, -34.3565369, 46.2026558, -76.1791382, 76.1023026
4: -28.7990532, 40.9032440, -32.2275734, 45.5036545, -74.3027039, 73.1308136

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5107103, upper bound: 96.5077763
time: 0.93 seconds

## Relational analysis of IS_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5107103, upper bound: 96.5472222
time: 0.97 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -45.9445267, 77.8574677, -33.9863777, 57.2567520, -103.2012711, 111.8438416
1: -50.2096825, 65.8146286, -37.1344414, 49.1948128, -99.4044800, 102.9490585
2: -51.4633865, 65.9640198, -38.0640297, 49.1163826, -100.5797729, 104.0280380
3: -59.2203903, 76.2339554, -43.8113251, 57.0491333, -116.2695236, 120.0452805
4: -54.2171478, 76.4686966, -40.4865379, 56.8555336, -111.0726700, 116.9552307

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5974363, upper bound: 96.5693809
time: 0.96 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5974363, upper bound: 96.5693809
time: 1.09 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -46.5680847, 78.4522629, -34.0134354, 57.2870483, -103.8551331, 112.4656982
1: -50.8739052, 66.4581451, -37.1634903, 49.2232819, -100.0971832, 103.6216278
2: -52.1392746, 66.6231461, -38.0933533, 49.1455421, -101.2848053, 104.7164993
3: -59.9515266, 76.9924545, -43.8434563, 57.0825615, -117.0340881, 120.8358994
4: -54.8835144, 77.2579498, -40.5155106, 56.8905067, -111.7740173, 117.7734604

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5827438, upper bound: 96.5614189
time: 0.78 seconds

## Relational analysis of IS_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5962316, upper bound: 96.5628326
time: 0.83 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -36.6144981, 60.9541779, -36.5990944, 61.4475403, -98.0620422, 97.5532684
1: -39.9822845, 52.5779991, -39.9751091, 52.7975769, -92.7798386, 92.5531082
2: -40.9669838, 52.4535255, -40.9522934, 52.6808434, -93.6478271, 93.4058228
3: -47.0317841, 61.1034889, -47.1351433, 61.3395424, -108.3713226, 108.2386322
4: -43.5240402, 60.8214378, -43.5151329, 61.0710258, -104.5950394, 104.3365707

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5999586, upper bound: 96.6001065
time: 0.91 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6035397, upper bound: 96.6035396
time: 0.87 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -36.6144981, 60.9541779, -37.1734772, 61.9913483, -98.6058502, 98.1276550
1: -39.9822845, 52.5779991, -40.5884628, 53.3791046, -93.3613739, 93.1664581
2: -40.9669838, 52.4535255, -41.5737152, 53.2809448, -94.2479095, 94.0272293
3: -47.0317841, 61.1034889, -47.8093529, 62.0262222, -109.0579910, 108.9128265
4: -43.5240402, 60.8214378, -44.1226883, 61.7857552, -105.3097610, 104.9441223

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6032123, upper bound: 96.6001065
time: 0.99 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6035397, upper bound: 96.6035396
time: 1.06 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -22.8007832, 43.5801544, -27.8737450, 49.8992538, -72.7000275, 71.4538956
1: -24.9963570, 35.9848976, -30.4715424, 41.7497940, -66.7461395, 66.4564209
2: -25.6877480, 35.6781197, -31.2517452, 41.5513115, -67.2390366, 66.9298553
3: -29.8977737, 41.6231422, -36.1432686, 48.3789749, -78.2767487, 77.7664108
4: -28.6988297, 40.8071556, -33.7890205, 47.7745819, -76.4733810, 74.5961685

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5140852, upper bound: 96.5418482
time: 0.96 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5140875, upper bound: 96.5597157
time: 1.19 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -46.6893158, 78.8251419, -35.7072029, 59.9363403, -106.6256485, 114.5323257
1: -51.0246201, 66.8172150, -38.9952698, 51.4772530, -102.5018768, 105.8124847
2: -52.2992477, 66.9673538, -39.9439240, 51.3954391, -103.6946869, 106.9112778
3: -60.1669884, 77.4223480, -45.9962387, 59.7308960, -119.8978882, 123.4185791
4: -55.1187363, 77.6611710, -42.3932495, 59.5483398, -114.6670532, 120.0544205

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5915879, upper bound: 96.6075919
time: 0.83 seconds

## Relational analysis of IS_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6105529, upper bound: 96.6105538
time: 1.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.15 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.5131282, upper bound: 96.5377879
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.5131282, upper bound: 96.5408059
IS_B1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.5107103, upper bound: 96.5077763
IS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.5107103, upper bound: 96.5472222
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.5974363, upper bound: 96.5693809
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.5974363, upper bound: 96.5693809
IS_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.5827438, upper bound: 96.5614189
IS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.5962316, upper bound: 96.5628326
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.5999586, upper bound: 96.6001065
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.6035397, upper bound: 96.6035396
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.6032123, upper bound: 96.6001065
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.6035397, upper bound: 96.6035396
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.5140852, upper bound: 96.5418482
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.5140875, upper bound: 96.5597157
IS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.5915879, upper bound: 96.6075919
IS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 4, lower bound: -96.6105529, upper bound: 96.6105538

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -20.9479790, 41.0590668, -25.1296062, 46.0035858, -66.9515686, 66.1886749
1: -23.0076408, 33.6860580, -27.4931812, 38.2287216, -61.2363625, 61.1792374
2: -23.6330490, 33.3669968, -28.2013359, 37.9926071, -61.6256561, 61.5683174
3: -27.5866184, 38.9304581, -32.6365852, 44.2879562, -71.8745728, 71.5670319
4: -26.7032413, 38.0687332, -30.6879864, 43.5646172, -70.2678604, 68.7567139

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5076091, upper bound: 96.5350327
time: 0.78 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5130275, upper bound: 96.5334598
time: 1.05 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5129557, upper bound: 96.5336187
time: 1.21 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -22.9442444, 42.7539444, -25.3594551, 45.7541428, -68.6983795, 68.1133881
1: -25.1168919, 35.6563644, -27.7241898, 38.2309570, -63.3478432, 63.3805542
2: -25.8165302, 35.3936996, -28.4438705, 38.0312805, -63.8477898, 63.8375702
3: -29.9805603, 41.2992325, -32.8748665, 44.3227692, -74.3033218, 74.1740875
4: -28.6963692, 40.5230637, -30.8441315, 43.6169472, -72.3133087, 71.3671951

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5076852, upper bound: 96.5377376
time: 0.84 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5100227, upper bound: 96.5325078
time: 0.85 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5100227, upper bound: 96.5336926
time: 0.87 seconds

## BFS IS instance: IS_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -22.8797913, 43.6451569, -43.5260849, 74.9641266, -97.8439102, 87.1712341
1: -25.0808849, 36.0795708, -47.6387863, 63.2178421, -88.2987289, 83.7183533
2: -25.7734623, 35.7657967, -48.8554382, 63.3163071, -89.0897675, 84.6212311
3: -29.9764824, 41.7457657, -56.3892250, 73.2049103, -103.1813965, 98.1349792
4: -28.7990532, 40.9032440, -51.6862946, 73.2945328, -102.0935822, 92.5895386

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5107103, upper bound: 96.5442473
time: 0.89 seconds

## Relational analysis of IS_B1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5097122, upper bound: 96.5472222
time: 0.93 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -45.9445267, 77.8574677, -33.3755226, 56.6895142, -102.6340332, 111.2329865
1: -50.2096825, 65.8146286, -36.4829445, 48.5845222, -98.7942047, 102.2975616
2: -51.4633865, 65.9640198, -37.4052658, 48.4866371, -99.9500275, 103.3692856
3: -59.2203903, 76.2339554, -43.0961990, 56.3302917, -115.5506821, 119.3301468
4: -54.2171478, 76.4686966, -39.8417473, 56.1063385, -110.3234863, 116.3104324

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5782236, upper bound: 96.5498287
time: 0.87 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5831600, upper bound: 96.5501802
time: 1.04 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -45.9445267, 77.8574677, -33.9720993, 57.2449226, -103.1894455, 111.8295593
1: -50.2096825, 65.8146286, -37.1194267, 49.1829109, -99.3925934, 102.9340439
2: -51.4633865, 65.9640198, -38.0490799, 49.1034050, -100.5667877, 104.0130844
3: -59.2203903, 76.2339554, -43.7954102, 57.0354614, -116.2558365, 120.0293655
4: -54.2171478, 76.4686966, -40.4729385, 56.8400345, -111.0571747, 116.9416199

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5782236, upper bound: 96.5498287
time: 0.81 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5831600, upper bound: 96.5501802
time: 0.98 seconds

## BFS IS instance: IS_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -45.6085243, 77.3161850, -33.0676231, 55.9470367, -101.5555573, 110.3838043
1: -49.8650436, 65.5379486, -36.1473236, 48.0375137, -97.9025574, 101.6852722
2: -51.0824089, 65.6959305, -37.0465393, 47.9542961, -99.0367050, 102.7424622
3: -58.8098869, 75.9158478, -42.6710129, 55.6963921, -114.5062637, 118.5868530
4: -53.8137894, 76.1261673, -39.4313316, 55.4707680, -109.2845535, 115.5574951

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_A1_B1

### Relational analysis result of IS_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5827438, upper bound: 96.5417214
time: 1.04 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2

### Relational analysis result of IS_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5827438, upper bound: 96.5614189
time: 0.80 seconds

## BFS IS instance: IS_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -44.9088860, 76.2383804, -33.2264557, 56.1898079, -101.0986710, 109.4648285
1: -49.1078072, 64.4969177, -36.3214073, 48.2449760, -97.3527832, 100.8183289
2: -50.3260956, 64.6249161, -37.2291527, 48.1537132, -98.4798126, 101.8540649
3: -57.9494629, 74.7412643, -42.8770599, 55.9367371, -113.8862000, 117.6183167
4: -53.0767555, 74.8945923, -39.6411667, 55.7292938, -108.8060455, 114.5357590

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_A2_A1

### Relational analysis result of IS_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5873140, upper bound: 96.5577000
time: 0.96 seconds

## Relational analysis of IS_B1_A2_A2_A2_A2

### Relational analysis result of IS_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5880512, upper bound: 96.5587938
time: 0.91 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -35.6075134, 59.5686722, -36.2676926, 61.2883148, -96.8958282, 95.8363647
1: -38.9039803, 51.3545532, -39.6585274, 52.8485870, -91.7525635, 91.0130768
2: -39.8581047, 51.2225609, -40.6028252, 52.6898232, -92.5479202, 91.8253708
3: -45.8003311, 59.6748161, -46.8099556, 61.3445587, -107.1448898, 106.4847717
4: -42.3867188, 59.3547211, -43.2034187, 61.0613937, -103.4481125, 102.5581207

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5803071, upper bound: 96.5941242
time: 1.02 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5803071, upper bound: 96.6001065
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -35.8111649, 59.8400269, -34.9046249, 59.0837555, -94.8949203, 94.7446442
1: -39.1246872, 51.5889320, -38.1617966, 50.7091293, -89.8338165, 89.7507172
2: -40.0861931, 51.4506760, -39.0921059, 50.5622253, -90.6483994, 90.5427704
3: -46.0533943, 59.9504356, -45.0599098, 58.9061089, -104.9595032, 105.0103378
4: -42.6336632, 59.6452103, -41.6373520, 58.5807266, -101.2143707, 101.2825394

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6001065, upper bound: 96.6032123
time: 1.09 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6001065, upper bound: 96.6035397
time: 1.12 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -35.6075134, 59.5686722, -36.8106537, 61.7748184, -97.3823242, 96.3793259
1: -38.9039803, 51.3545532, -40.2371979, 53.3699532, -92.2739258, 91.5917511
2: -39.8581047, 51.2225609, -41.1873512, 53.2342072, -93.0922928, 92.4098969
3: -45.8003311, 59.6748161, -47.4439087, 61.9625893, -107.7629242, 107.1187057
4: -42.3867188, 59.3547211, -43.7690887, 61.7081261, -104.0948486, 103.1238098

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B2_B1_B1

### Relational analysis result of IS_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5348492, upper bound: 96.5106538
time: 1.64 seconds

## Relational analysis of IS_B2_A1_B2_B1_B2

### Relational analysis result of IS_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6007353, upper bound: 96.5909336
time: 1.29 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -35.8111649, 59.8400269, -35.5050240, 59.6647072, -95.4758759, 95.3450317
1: -39.1246872, 51.5889320, -38.8045807, 51.3255806, -90.4502716, 90.3935089
2: -40.0861931, 51.4506760, -39.7425194, 51.1979942, -91.2841644, 91.1931915
3: -46.0533943, 59.9504356, -45.7699165, 59.6335678, -105.6869659, 105.7203445
4: -42.6336632, 59.6452103, -42.2741013, 59.3359261, -101.9695892, 101.9192810

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B2_B2_B1

### Relational analysis result of IS_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5371544, upper bound: 96.5139810
time: 1.07 seconds

## Relational analysis of IS_B2_A1_B2_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6031407, upper bound: 96.6098577
time: 1.95 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -22.8007832, 43.5801544, -27.3139286, 49.4381790, -72.2389603, 70.8940811
1: -24.9963570, 35.9848976, -29.8832397, 41.2416725, -66.2380219, 65.8681335
2: -25.6877480, 35.6781197, -30.6496162, 41.0206299, -66.7083588, 66.3277206
3: -29.8977737, 41.6231422, -35.5108566, 47.7742958, -77.6720734, 77.1340027
4: -28.6988297, 40.8071556, -33.2124634, 47.1502342, -75.8490601, 74.0195999

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5140837, upper bound: 96.5409753
time: 0.91 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5140034, upper bound: 96.5418482
time: 0.80 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -22.8007832, 43.5801544, -27.8374691, 49.8666267, -72.6674118, 71.4176254
1: -24.9963570, 35.9848976, -30.4338226, 41.7187805, -66.7151337, 66.4187088
2: -25.6877480, 35.6781197, -31.2134933, 41.5185547, -67.2062836, 66.8916092
3: -29.8977737, 41.6231422, -36.1038246, 48.3419800, -78.2397461, 77.7269669
4: -28.6988297, 40.8071556, -33.7558441, 47.7354279, -76.4342575, 74.5629883

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5140859, upper bound: 96.5594587
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5140054, upper bound: 96.5563500
time: 0.99 seconds

## BFS IS instance: IS_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -45.8310547, 77.8317261, -34.7175140, 58.5800209, -104.4110718, 112.5492249
1: -50.1351471, 66.0440826, -37.9385490, 50.2798843, -100.4150238, 103.9825974
2: -51.3563423, 66.1805954, -38.8546677, 50.1906929, -101.5470352, 105.0352631
3: -59.1700516, 76.5192490, -44.7966118, 58.3304520, -117.5005035, 121.3158569
4: -54.1662445, 76.7004013, -41.2716141, 58.1169739, -112.2832184, 117.9720154

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5893527, upper bound: 96.5893536
time: 0.86 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5869480, upper bound: 96.6075919
time: 0.83 seconds

## BFS IS instance: IS_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -44.9828796, 76.5656128, -34.8912392, 58.7923737, -103.7752533, 111.4568329
1: -49.2089272, 64.8141022, -38.1233673, 50.4685211, -99.6774445, 102.9374619
2: -50.4367409, 64.9280014, -39.0478668, 50.3715515, -100.8082886, 103.9758682
3: -58.1163177, 75.1221619, -44.9994965, 58.5512848, -116.6676025, 120.1216431
4: -53.2640343, 75.2480469, -41.4901924, 58.3480682, -111.6120911, 116.7382355

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5980579, upper bound: 96.5976753
time: 0.95 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5981862, upper bound: 96.5981873
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.18 seconds
IS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5130275, upper bound: 96.5334598
IS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5129557, upper bound: 96.5336187
IS_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5100227, upper bound: 96.5325078
IS_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5100227, upper bound: 96.5336926
IS_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5107103, upper bound: 96.5442473
IS_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5097122, upper bound: 96.5472222
IS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5782236, upper bound: 96.5498287
IS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5831600, upper bound: 96.5501802
IS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5782236, upper bound: 96.5498287
IS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5831600, upper bound: 96.5501802
IS_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5827438, upper bound: 96.5417214
IS_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5827438, upper bound: 96.5614189
IS_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5873140, upper bound: 96.5577000
IS_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5880512, upper bound: 96.5587938
IS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5803071, upper bound: 96.5941242
IS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5803071, upper bound: 96.6001065
IS_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.6001065, upper bound: 96.6032123
IS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.6001065, upper bound: 96.6035397
IS_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5348492, upper bound: 96.5106538
IS_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.6007353, upper bound: 96.5909336
IS_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5371544, upper bound: 96.5139810
IS_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.6031407, upper bound: 96.6098577
IS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5140837, upper bound: 96.5409753
IS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5140034, upper bound: 96.5418482
IS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5140859, upper bound: 96.5594587
IS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5140054, upper bound: 96.5563500
IS_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5893527, upper bound: 96.5893536
IS_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5869480, upper bound: 96.6075919
IS_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5980579, upper bound: 96.5976753
IS_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.18
Output dim: 4, lower bound: -96.5981862, upper bound: 96.5981873

## BFS IS instance: IS_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -20.1299171, 39.9241028, -24.5777016, 45.2647972, -65.3947067, 64.5018005
1: -22.1376400, 32.6815071, -26.8915253, 37.6648712, -59.8025093, 59.5730324
2: -22.7294083, 32.3638535, -27.5909843, 37.4319077, -60.1613083, 59.9548302
3: -26.5982819, 37.7569122, -31.9388676, 43.5820999, -70.1803818, 69.6957779
4: -25.7783623, 36.8808212, -29.9685726, 42.8679161, -68.6462708, 66.8493881

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5075729, upper bound: 96.5306277
time: 0.86 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5070845, upper bound: 96.5314121
time: 1.41 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5070845, upper bound: 96.5334598
time: 0.90 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -20.2531834, 40.0984039, -23.7688599, 44.0332222, -64.2863998, 63.8672638
1: -22.2652321, 32.8131752, -26.0239010, 36.4522209, -58.7174492, 58.8370667
2: -22.8622246, 32.4924393, -26.6872330, 36.2015305, -59.0637550, 59.1796722
3: -26.7360382, 37.9105568, -30.9282074, 42.2300758, -68.9661102, 68.8387527
4: -25.9140892, 37.0378036, -29.1586533, 41.4593658, -67.3734436, 66.1964493

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5074505, upper bound: 96.5307936
time: 1.06 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5062821, upper bound: 96.5317119
time: 0.95 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5062821, upper bound: 96.5336187
time: 0.97 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -22.1215477, 41.6306953, -24.5328407, 44.5526085, -66.6741486, 66.1635284
1: -24.2266273, 34.6650887, -26.8320293, 37.1714363, -61.3980637, 61.4971085
2: -24.8897705, 34.4419937, -27.5237350, 36.9778366, -61.8676071, 61.9657288
3: -28.9556522, 40.0951195, -31.8424053, 43.0789566, -72.0345993, 71.9375229
4: -27.6592312, 39.3680763, -29.8841019, 42.3525009, -70.0117188, 69.2521820

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B1_A2_A1_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5046481, upper bound: 96.5296364
time: 0.78 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A2_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5100227, upper bound: 96.5325078
time: 0.78 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_B2

### Relational analysis result of IS_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5100227, upper bound: 96.5325078
time: 0.89 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -21.5203190, 40.7735519, -24.6046982, 44.6780434, -66.1983490, 65.3782501
1: -23.5980339, 33.8506927, -26.9119492, 37.2526245, -60.8506584, 60.7626419
2: -24.2401485, 33.5712433, -27.6063824, 37.0409088, -61.2810593, 61.1776161
3: -28.2352829, 39.2045326, -31.9348812, 43.1895790, -71.4248657, 71.1394119
4: -27.0703430, 38.3959808, -29.9971809, 42.4623299, -69.5326614, 68.3931503

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B1_A2_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5046481, upper bound: 96.5308655
time: 0.78 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A2_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5129367, upper bound: 96.5336926
time: 1.13 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5129367, upper bound: 96.5336926
time: 1.07 seconds

## BFS IS instance: IS_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -20.9696808, 41.0813789, -42.7052994, 73.7626953, -94.7323761, 83.7866669
1: -23.0300903, 33.7072868, -46.7348366, 62.1171074, -85.1471939, 80.4421234
2: -23.6562309, 33.3887367, -47.9329491, 62.2049255, -85.8611603, 81.3216705
3: -27.6105652, 38.9555397, -55.3195610, 71.9182129, -99.5287781, 94.2751007
4: -26.7258835, 38.0942039, -50.7400246, 71.9723129, -98.6981964, 88.8342285

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B2_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5090381, upper bound: 96.5409915
time: 1.24 seconds

## Relational analysis of IS_B1_A1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5143586, upper bound: 96.5382129
time: 0.98 seconds

## Relational analysis of IS_B1_A1_B2_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5143157, upper bound: 96.5410738
time: 0.79 seconds

## BFS IS instance: IS_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -22.9650612, 42.7756310, -42.6973228, 73.2336578, -96.1987152, 85.4729462
1: -25.1384754, 35.6768875, -46.7225685, 61.8569679, -86.9954376, 82.3994598
2: -25.8388519, 35.4146194, -47.9137268, 61.9716110, -87.8104553, 83.3283386
3: -30.0038223, 41.3234825, -55.2999954, 71.6293488, -101.6331711, 96.6234741
4: -28.7185020, 40.5477943, -50.6659622, 71.7472458, -100.4657440, 91.2137604

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_A1_B2_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5090381, upper bound: 96.5438489
time: 1.05 seconds

## Relational analysis of IS_B1_A1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_B2_A2_B1

### Relational analysis result of IS_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5143483, upper bound: 96.5438181
time: 0.80 seconds

## Relational analysis of IS_B1_A1_B2_B2_A2_B2

### Relational analysis result of IS_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5142768, upper bound: 96.5423893
time: 1.06 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -45.0676804, 76.8108673, -32.4971581, 55.3968658, -100.4645462, 109.3080292
1: -49.2871666, 64.9865723, -35.5347824, 47.4449043, -96.7320709, 100.5213318
2: -50.4956169, 65.1284637, -36.4273758, 47.3435593, -97.8391724, 101.5558395
3: -58.1703835, 75.2672882, -41.9900246, 54.9996376, -113.1699905, 117.2573090
4: -53.2430000, 75.4447403, -38.8223991, 54.7437782, -107.9867554, 114.2671356

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5782236, upper bound: 96.5498287
time: 1.02 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5782236, upper bound: 96.5498287
time: 1.04 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -44.2982101, 75.6494598, -32.6471977, 55.6325989, -99.9308014, 108.2966537
1: -48.4563103, 63.8612862, -35.7008438, 47.6438217, -96.1001282, 99.5621262
2: -49.6631889, 63.9742126, -36.6007347, 47.5350456, -97.1982346, 100.5749359
3: -57.2300262, 73.9919586, -42.1881752, 55.2287025, -112.4587173, 116.1801300
4: -52.4242554, 74.1150055, -39.0196915, 54.9942818, -107.4185257, 113.1346970

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5649983, upper bound: 96.5427260
time: 1.05 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5751293, upper bound: 96.5468618
time: 0.98 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -45.0676804, 76.8108673, -33.0305710, 55.9075737, -100.9752502, 109.8414383
1: -49.2871666, 64.9865723, -36.1075249, 47.9995575, -97.2867203, 101.0940933
2: -50.4956169, 65.1284637, -37.0065880, 47.9148178, -98.4104309, 102.1350555
3: -58.1703835, 75.2672882, -42.6271019, 55.6520729, -113.8224487, 117.8943710
4: -53.2430000, 75.4447403, -39.3923569, 55.4235077, -108.6664963, 114.8370819

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5803154, upper bound: 96.5612990
time: 0.89 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5782236, upper bound: 96.5612990
time: 0.91 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -44.2982101, 75.6494598, -33.1908722, 56.1524582, -100.4506683, 108.8403320
1: -48.4563103, 63.8612862, -36.2832375, 48.2091637, -96.6654739, 100.1445236
2: -49.6631889, 63.9742126, -37.1908493, 48.1163712, -97.7795563, 101.1650467
3: -57.2300262, 73.9919586, -42.8349686, 55.8950119, -113.1250305, 116.8269272
4: -52.4242554, 74.1150055, -39.6040001, 55.6844406, -108.1086960, 113.7190094

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B2_A2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5684981, upper bound: 96.5513965
time: 1.02 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5794202, upper bound: 96.5579226
time: 1.11 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -45.6085243, 77.3161850, -33.2625771, 56.4974136, -102.1059341, 110.5787659
1: -49.8650436, 65.5379486, -36.3766632, 48.6481018, -98.5131454, 101.9146118
2: -51.0824089, 65.6959305, -37.2786407, 48.5441818, -99.6265869, 102.9745712
3: -58.8098869, 75.9158478, -42.9615517, 56.4075127, -115.2173996, 118.8773956
4: -53.8137894, 76.1261673, -39.7078362, 56.1503792, -109.9641571, 115.8339920

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_A2_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5743180, upper bound: 96.5436263
time: 1.01 seconds

## Relational analysis of IS_B1_A2_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5799943, upper bound: 96.5586024
time: 0.88 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -45.6085243, 77.3161850, -32.5994263, 55.2373810, -100.8458862, 109.9156113
1: -49.8650436, 65.5379486, -35.6437683, 47.4086227, -97.2736664, 101.1817093
2: -51.0824089, 65.6959305, -36.5337753, 47.3110008, -98.3934097, 102.2296906
3: -58.8098869, 75.9158478, -42.0838890, 54.9588013, -113.7686768, 117.9997406
4: -53.8137894, 76.1261673, -38.9216118, 54.7403336, -108.5541229, 115.0477753

Time for backsubstitution: 2.11 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=108.20734405517578
rel_dist={4: [-96.6182045307215, 96.61820453072153]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1115.12 seconds
