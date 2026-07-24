## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_1.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 843.1946849690161


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-204.6976013, 696.8228149, -204.6976013, 696.8228149, -901.5203857, 901.5203857)
1: (-334.7795410, 851.1083374, -334.7795410, 851.1083374, -1185.8878174, 1185.8878174)
2: (-233.0683441, 900.9995117, -233.0683441, 900.9995117, -1134.0678711, 1134.0678711)
3: (-594.6608887, 866.9309082, -594.6608887, 866.9309082, -1461.5917969, 1461.5917969)
4: (-370.8992920, 924.7813721, -370.8992920, 924.7813721, -1295.6806641, 1295.6806641)

## BASE Result
execution time: IAR + LP analysis = 2.15 + 2.09 = 4.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -843.2117555, upper bound: 843.2117555


# Binary Search by BASE starts (time budget: 1195.76 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=901.5203857421875
rel_dist={0: [-843.2117555388025, 843.2117555388029]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=901.5203857421875
rel_dist={0: [-843.2117464329822, 843.2117464329822]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=901.5203857421875
rel_dist={0: [-843.2116141859327, 843.2116141859326]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=901.5203857421875
rel_dist={0: [-843.2113518990857, 843.2113518990857]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=901.5203857421875
rel_dist={0: [-843.2111420440377, 843.2111420440376]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=901.5203857421875
rel_dist={0: [-843.2109964317544, 843.2109964317542]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=901.5203857421875
rel_dist={0: [-843.2108890178715, 843.2108890178715]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=901.5203857421875
rel_dist={0: [-843.2108286282617, 843.2108286282617]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=901.5203857421875
rel_dist={0: [-843.210795750624, 843.2107957506241]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=901.5203857421875
rel_dist={0: [-843.21077796134, 843.2107779613402]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=901.5203857421875
rel_dist={0: [-843.2107690666991, 843.2107690666994]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=901.5203857421875
rel_dist={0: [-843.2107645471314, 843.210764547131]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=901.5203857421875
rel_dist={0: [-843.2107622632411, 843.2107622632411]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=901.5203857421875
rel_dist={0: [-843.2107611213063, 843.2107611213064]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=901.5203857421875
rel_dist={0: [-843.2107605503594, 843.2107605503595]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=901.5203857421875
rel_dist={0: [-843.21076026523, 843.210760264925]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=901.5203857421875
rel_dist={0: [-843.2107601224682, 843.2107601222826]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=901.5203857421875
rel_dist={0: [-843.2107600512198, 843.210760052501]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=901.5203857421875
rel_dist={0: [-843.2107600182171, 843.2107600215168]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=901.5203857421875
rel_dist={0: [-843.2107600027682, 843.2107600018253]}

## Binary Search Result
Binary search time: 85.06 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1110.70 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2098771, upper bound: 843.2088945
time: 0.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078139, upper bound: 843.2078139
time: 0.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -843.2098771, upper bound: 843.2088945
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -843.2078139, upper bound: 843.2078139

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -188.3924713, 641.2564087, -204.6976013, 696.8228149, -885.2152100, 845.9539795
1: -307.3268738, 783.6063843, -334.7795410, 851.1083374, -1158.4351807, 1118.3857422
2: -214.6040497, 829.6708374, -233.0683441, 900.9995117, -1115.6035156, 1062.7391357
3: -547.3501587, 798.2861328, -594.6608887, 866.9309082, -1414.2810059, 1392.9470215
4: -341.8905029, 851.8900757, -370.8992920, 924.7813721, -1266.6718750, 1222.7893066

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078139, upper bound: 843.2078139
time: 0.67 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078139, upper bound: 843.2078139
time: 0.69 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -191.9384003, 653.1663818, -204.6976013, 696.8228149, -888.7612305, 857.8639526
1: -314.3370667, 797.7437744, -334.7795410, 851.1083374, -1165.4454346, 1132.5229492
2: -218.4419098, 843.8818970, -233.0683441, 900.9995117, -1119.4414062, 1076.9501953
3: -557.8200684, 812.7958374, -594.6608887, 866.9309082, -1424.7509766, 1407.4567871
4: -347.6372986, 866.6468506, -370.8992920, 924.7813721, -1272.4184570, 1237.5461426

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078139, upper bound: 843.2078139
time: 0.77 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078139, upper bound: 843.2078139
time: 0.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.82 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.82
Output dim: 0, lower bound: -843.2078139, upper bound: 843.2078139
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.82
Output dim: 0, lower bound: -843.2078139, upper bound: 843.2078139
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.82
Output dim: 0, lower bound: -843.2078139, upper bound: 843.2078139
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.82
Output dim: 0, lower bound: -843.2078139, upper bound: 843.2078139

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -188.3924713, 641.2564087, -188.3924713, 641.2564087, -829.6488647, 829.6488647
1: -307.3268738, 783.6063843, -307.3268738, 783.6063843, -1090.9332275, 1090.9332275
2: -214.6040497, 829.6708374, -214.6040497, 829.6708374, -1044.2749023, 1044.2749023
3: -547.3501587, 798.2861328, -547.3501587, 798.2861328, -1345.6362305, 1345.6362305
4: -341.8905029, 851.8900757, -341.8905029, 851.8900757, -1193.7805176, 1193.7805176

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2090113, upper bound: 843.2078449
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2093633, upper bound: 843.2083725
time: 0.71 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -188.3924713, 641.2564087, -191.9384003, 653.1663818, -841.5588379, 833.1948242
1: -307.3268738, 783.6063843, -314.3370667, 797.7437744, -1105.0705566, 1097.9434814
2: -214.6040497, 829.6708374, -218.4419098, 843.8818970, -1058.4859619, 1048.1127930
3: -547.3501587, 798.2861328, -557.8200684, 812.7958374, -1360.1459961, 1356.1062012
4: -341.8905029, 851.8900757, -347.6372986, 866.6468506, -1208.5373535, 1199.5273438

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2090113, upper bound: 843.2078449
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2093633, upper bound: 843.2083725
time: 0.85 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -191.9384003, 653.1663818, -188.3924713, 641.2564087, -833.1948242, 841.5588379
1: -314.3370667, 797.7437744, -307.3268738, 783.6063843, -1097.9434814, 1105.0705566
2: -218.4419098, 843.8818970, -214.6040497, 829.6708374, -1048.1127930, 1058.4859619
3: -557.8200684, 812.7958374, -547.3501587, 798.2861328, -1356.1062012, 1360.1459961
4: -347.6372986, 866.6468506, -341.8905029, 851.8900757, -1199.5273438, 1208.5373535

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2068924, upper bound: 843.2067674
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072349, upper bound: 843.2072349
time: 0.72 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -191.9384003, 653.1663818, -191.9384003, 653.1663818, -845.1047974, 845.1047974
1: -314.3370667, 797.7437744, -314.3370667, 797.7437744, -1112.0806885, 1112.0806885
2: -218.4419098, 843.8818970, -218.4419098, 843.8818970, -1062.3238525, 1062.3238525
3: -557.8200684, 812.7958374, -557.8200684, 812.7958374, -1370.6158447, 1370.6158447
4: -347.6372986, 866.6468506, -347.6372986, 866.6468506, -1214.2841797, 1214.2841797

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2068924, upper bound: 843.2067674
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072349, upper bound: 843.2072349
time: 0.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.22 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -843.2090113, upper bound: 843.2078449
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -843.2093633, upper bound: 843.2083725
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -843.2090113, upper bound: 843.2078449
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -843.2093633, upper bound: 843.2083725
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -843.2068924, upper bound: 843.2067674
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -843.2072349, upper bound: 843.2072349
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -843.2068924, upper bound: 843.2067674
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 0, lower bound: -843.2072349, upper bound: 843.2072349

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -184.1930847, 626.9782715, -188.3924713, 641.2564087, -825.4493408, 815.3707275
1: -300.4424438, 766.2285767, -307.3268738, 783.6063843, -1084.0488281, 1073.5554199
2: -209.7961884, 811.2807007, -214.6040497, 829.6708374, -1039.4670410, 1025.8847656
3: -535.2235718, 780.4590454, -547.3501587, 798.2861328, -1333.5097656, 1327.8092041
4: -334.2570190, 833.0983276, -341.8905029, 851.8900757, -1186.1470947, 1174.9887695

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2086150, upper bound: 843.2099345
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2086653, upper bound: 843.2091117
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -185.4147034, 630.6639404, -188.3924713, 641.2564087, -826.6711426, 819.0563965
1: -302.0009155, 770.8518677, -307.3268738, 783.6063843, -1085.6072998, 1078.1787109
2: -211.2521362, 816.2529297, -214.6040497, 829.6708374, -1040.9229736, 1030.8569336
3: -538.4077759, 785.4230957, -547.3501587, 798.2861328, -1336.6937256, 1332.7731934
4: -336.6280823, 838.2955933, -341.8905029, 851.8900757, -1188.5180664, 1180.1860352

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096367, upper bound: 843.2105097
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096870, upper bound: 843.2096870
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -184.1930847, 626.9782715, -191.9384003, 653.1663818, -837.3593140, 818.9166870
1: -300.4424438, 766.2285767, -314.3370667, 797.7437744, -1098.1860352, 1080.5656738
2: -209.7961884, 811.2807007, -218.4419098, 843.8818970, -1053.6781006, 1029.7226562
3: -535.2235718, 780.4590454, -557.8200684, 812.7958374, -1348.0194092, 1338.2790527
4: -334.2570190, 833.0983276, -347.6372986, 866.6468506, -1200.9038086, 1180.7355957

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2075031, upper bound: 843.2077972
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2085634, upper bound: 843.2075024
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2085634, upper bound: 843.2078449
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -185.4147034, 630.6639404, -191.9384003, 653.1663818, -838.5810547, 822.6023560
1: -302.0009155, 770.8518677, -314.3370667, 797.7437744, -1099.7445068, 1085.1887207
2: -211.2521362, 816.2529297, -218.4419098, 843.8818970, -1055.1340332, 1034.6947021
3: -538.4077759, 785.4230957, -557.8200684, 812.7958374, -1351.2034912, 1343.2431641
4: -336.6280823, 838.2955933, -347.6372986, 866.6468506, -1203.2749023, 1185.9328613

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2085248, upper bound: 843.2083725
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2088958, upper bound: 843.2080300
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2088958, upper bound: 843.2083725
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -188.3924713, 641.2564087, -829.7315674, 829.6968994
1: -308.6676941, 783.1837769, -307.3268738, 783.6063843, -1092.2739258, 1090.5106201
2: -214.4147797, 828.5700073, -214.6040497, 829.6708374, -1044.0855713, 1043.1740723
3: -547.7684326, 797.9298096, -547.3501587, 798.2861328, -1346.0541992, 1345.2800293
4: -341.2081909, 850.9271851, -341.8905029, 851.8900757, -1193.0981445, 1192.8176270

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2075024, upper bound: 843.2085634
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2075024, upper bound: 843.2088958
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -188.3924713, 641.2564087, -828.3795776, 824.6336670
1: -305.6914673, 777.5458984, -307.3268738, 783.6063843, -1089.2978516, 1084.8728027
2: -213.0952911, 822.5731201, -214.6040497, 829.6708374, -1042.7661133, 1037.1771240
3: -543.3413086, 792.1798706, -547.3501587, 798.2861328, -1341.6274414, 1339.5300293
4: -339.2044983, 844.9370728, -341.8905029, 851.8900757, -1191.0946045, 1186.8275146

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078449, upper bound: 843.2090113
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078449, upper bound: 843.2093633
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -191.9384003, 653.1663818, -841.6414795, 833.2428589
1: -308.6676941, 783.1837769, -314.3370667, 797.7437744, -1106.4111328, 1097.5208740
2: -214.4147797, 828.5700073, -218.4419098, 843.8818970, -1058.2966309, 1047.0119629
3: -547.7684326, 797.9298096, -557.8200684, 812.7958374, -1360.5639648, 1355.7497559
4: -341.2081909, 850.9271851, -347.6372986, 866.6468506, -1207.8549805, 1198.5644531

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2064249, upper bound: 843.2064249
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2064249, upper bound: 843.2067674
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -191.9384003, 653.1663818, -840.2895508, 828.1796265
1: -305.6914673, 777.5458984, -314.3370667, 797.7437744, -1103.4350586, 1091.8829346
2: -213.0952911, 822.5731201, -218.4419098, 843.8818970, -1056.9771729, 1041.0150146
3: -543.3413086, 792.1798706, -557.8200684, 812.7958374, -1356.1372070, 1349.9998779
4: -339.2044983, 844.9370728, -347.6372986, 866.6468506, -1205.8513184, 1192.5740967

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067674, upper bound: 843.2068924
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067674, upper bound: 843.2072349
time: 0.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.06 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2086150, upper bound: 843.2099345
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2086653, upper bound: 843.2091117
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2096367, upper bound: 843.2105097
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2096870, upper bound: 843.2096870
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2085634, upper bound: 843.2075024
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2085634, upper bound: 843.2078449
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2088958, upper bound: 843.2080300
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2088958, upper bound: 843.2083725
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2075024, upper bound: 843.2085634
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2075024, upper bound: 843.2088958
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2078449, upper bound: 843.2090113
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2078449, upper bound: 843.2093633
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2064249, upper bound: 843.2064249
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2064249, upper bound: 843.2067674
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2067674, upper bound: 843.2068924
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 0, lower bound: -843.2067674, upper bound: 843.2072349

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -184.1930847, 626.9782715, -171.6107483, 583.5519409, -767.7448730, 798.5889893
1: -300.4424438, 766.2285767, -280.5059509, 713.3177490, -1013.7601929, 1046.7344971
2: -209.7961884, 811.2807007, -195.5651093, 754.9420776, -964.7381592, 1006.8458252
3: -535.2235718, 780.4590454, -499.1630554, 726.9379272, -1262.1614990, 1279.6220703
4: -334.2570190, 833.0983276, -311.6126404, 775.2189941, -1109.4760742, 1144.7109375

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2086150, upper bound: 843.2091117
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2086150, upper bound: 843.2091117
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -183.7432556, 625.4803467, -208.0431061, 707.0399780, -890.7832031, 833.5233765
1: -299.7053223, 764.3865967, -342.9703064, 862.9215698, -1162.6269531, 1107.3569336
2: -209.2822418, 809.3569946, -237.0101166, 913.2547607, -1122.5369873, 1046.3670654
3: -533.9287109, 778.5666504, -607.4349976, 879.5144653, -1413.4431152, 1386.0017090
4: -333.4377136, 831.1161499, -376.8020630, 938.5618896, -1271.9993896, 1207.9180908

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2086653, upper bound: 843.2091117
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2086653, upper bound: 843.2091117
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -185.4147034, 630.6639404, -171.6107483, 583.5519409, -768.9666748, 802.2746582
1: -302.0009155, 770.8518677, -280.5059509, 713.3177490, -1015.3186646, 1051.3577881
2: -211.2521362, 816.2529297, -195.5651093, 754.9420776, -966.1942139, 1011.8180542
3: -538.4077759, 785.4230957, -499.1630554, 726.9379272, -1265.3455811, 1284.5861816
4: -336.6280823, 838.2955933, -311.6126404, 775.2189941, -1111.8468018, 1149.9082031

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096367, upper bound: 843.2096367
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096367, upper bound: 843.2096870
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -185.0043488, 629.2900391, -208.0431061, 707.0399780, -892.0443115, 837.3330688
1: -301.3310852, 769.1598511, -342.9703064, 862.9215698, -1164.2526855, 1112.1300049
2: -210.7831726, 814.4873657, -237.0101166, 913.2547607, -1124.0379639, 1051.4974365
3: -537.2214355, 783.6879883, -607.4349976, 879.5144653, -1416.7355957, 1391.1230469
4: -335.8796997, 836.4770508, -376.8020630, 938.5618896, -1274.4412842, 1213.2790527

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096870, upper bound: 843.2096367
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096870, upper bound: 843.2096870
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -184.1930847, 626.9782715, -188.4751282, 641.3044434, -825.4973755, 815.4533691
1: -300.4424438, 766.2285767, -308.6676941, 783.1837769, -1083.6262207, 1074.8962402
2: -209.7961884, 811.2807007, -214.4147797, 828.5700073, -1038.3662109, 1025.6954346
3: -535.2235718, 780.4590454, -547.7684326, 797.9298096, -1333.1533203, 1328.2274170
4: -334.2570190, 833.0983276, -341.2081909, 850.9271851, -1185.1842041, 1174.3063965

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2085426, upper bound: 843.2074550
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072649, upper bound: 843.2071630
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -184.1930847, 626.9782715, -187.1232147, 636.2412109, -820.4341431, 814.1014404
1: -300.4424438, 766.2285767, -305.6914673, 777.5458984, -1077.9882812, 1071.9200439
2: -209.7961884, 811.2807007, -213.0952911, 822.5731201, -1032.3692627, 1024.3759766
3: -535.2235718, 780.4590454, -543.3413086, 792.1798706, -1327.4034424, 1323.8002930
4: -334.2570190, 833.0983276, -339.2044983, 844.9370728, -1179.1940918, 1172.3028564

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2085426, upper bound: 843.2077645
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072649, upper bound: 843.2074725
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -185.4147034, 630.6639404, -188.4751282, 641.3044434, -826.7191162, 819.1390381
1: -302.0009155, 770.8518677, -308.6676941, 783.1837769, -1085.1846924, 1079.5192871
2: -211.2521362, 816.2529297, -214.4147797, 828.5700073, -1039.8221436, 1030.6677246
3: -538.4077759, 785.4230957, -547.7684326, 797.9298096, -1336.3374023, 1333.1914062
4: -336.6280823, 838.2955933, -341.2081909, 850.9271851, -1187.5551758, 1179.5036621

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2088594, upper bound: 843.2079844
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2082051, upper bound: 843.2078292
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -185.4147034, 630.6639404, -187.1232147, 636.2412109, -821.6558838, 817.7871094
1: -302.0009155, 770.8518677, -305.6914673, 777.5458984, -1079.5468750, 1076.5432129
2: -211.2521362, 816.2529297, -213.0952911, 822.5731201, -1033.8251953, 1029.3481445
3: -538.4077759, 785.4230957, -543.3413086, 792.1798706, -1330.5875244, 1328.7644043
4: -336.6280823, 838.2955933, -339.2044983, 844.9370728, -1181.5648193, 1177.5001221

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2088594, upper bound: 843.2082939
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2082052, upper bound: 843.2081387
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -184.1930847, 626.9782715, -815.4533691, 825.4973755
1: -308.6676941, 783.1837769, -300.4424438, 766.2285767, -1074.8962402, 1083.6262207
2: -214.4147797, 828.5700073, -209.7961884, 811.2807007, -1025.6954346, 1038.3662109
3: -547.7684326, 797.9298096, -535.2235718, 780.4590454, -1328.2274170, 1333.1533203
4: -341.2081909, 850.9271851, -334.2570190, 833.0983276, -1174.3063965, 1185.1842041

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074547, upper bound: 843.2070332
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071878, upper bound: 843.2070303
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2072649
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -185.4147034, 630.6639404, -819.1390381, 826.7191162
1: -308.6676941, 783.1837769, -302.0009155, 770.8518677, -1079.5192871, 1085.1846924
2: -214.4147797, 828.5700073, -211.2521362, 816.2529297, -1030.6676025, 1039.8221436
3: -547.7684326, 797.9298096, -538.4077759, 785.4230957, -1333.1914062, 1336.3374023
4: -341.2081909, 850.9271851, -336.6280823, 838.2955933, -1179.5036621, 1187.5551758

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074547, upper bound: 843.2070332
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071878, upper bound: 843.2079705
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2072649
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -184.1930847, 626.9782715, -814.1014404, 820.4341431
1: -305.6914673, 777.5458984, -300.4424438, 766.2285767, -1071.9200439, 1077.9882812
2: -213.0952911, 822.5731201, -209.7961884, 811.2807007, -1024.3759766, 1032.3692627
3: -543.3413086, 792.1798706, -535.2235718, 780.4590454, -1323.8002930, 1327.4034424
4: -339.2044983, 844.9370728, -334.2570190, 833.0983276, -1172.3028564, 1179.1940918

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077972, upper bound: 843.2075031
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074014, upper bound: 843.2074630
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074724, upper bound: 843.2077407
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -185.4147034, 630.6639404, -817.7871094, 821.6558838
1: -305.6914673, 777.5458984, -302.0009155, 770.8518677, -1076.5432129, 1079.5468750
2: -213.0952911, 822.5731201, -211.2521362, 816.2529297, -1029.3481445, 1033.8251953
3: -543.3413086, 792.1798706, -538.4077759, 785.4230957, -1328.7644043, 1330.5875244
4: -339.2044983, 844.9370728, -336.6280823, 838.2955933, -1177.5001221, 1181.5648193

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077972, upper bound: 843.2083136
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074014, upper bound: 843.2082965
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074725, upper bound: 843.2085883
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -188.4751282, 641.3044434, -829.7795410, 829.7795410
1: -308.6676941, 783.1837769, -308.6676941, 783.1837769, -1091.8514404, 1091.8514404
2: -214.4147797, 828.5700073, -214.4147797, 828.5700073, -1042.9847412, 1042.9847412
3: -547.7684326, 797.9298096, -547.7684326, 797.9298096, -1345.6978760, 1345.6978760
4: -341.2081909, 850.9271851, -341.2081909, 850.9271851, -1192.1352539, 1192.1352539

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063678, upper bound: 843.2061084
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063430, upper bound: 843.2063430
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -187.1232147, 636.2412109, -824.7163086, 828.4276123
1: -308.6676941, 783.1837769, -305.6914673, 777.5458984, -1086.2135010, 1088.8752441
2: -214.4147797, 828.5700073, -213.0952911, 822.5731201, -1036.9879150, 1041.6652832
3: -547.7684326, 797.9298096, -543.3413086, 792.1798706, -1339.9479980, 1341.2711182
4: -341.2081909, 850.9271851, -339.2044983, 844.9370728, -1186.1448975, 1190.1317139

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063678, upper bound: 843.2061084
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063430, upper bound: 843.2063430
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -188.4751282, 641.3044434, -828.4276123, 824.7163086
1: -305.6914673, 777.5458984, -308.6676941, 783.1837769, -1088.8752441, 1086.2135010
2: -213.0952911, 822.5731201, -214.4147797, 828.5700073, -1041.6652832, 1036.9879150
3: -543.3413086, 792.1798706, -547.7684326, 797.9298096, -1341.2711182, 1339.9479980
4: -339.2044983, 844.9370728, -341.2081909, 850.9271851, -1190.1317139, 1186.1448975

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065814, upper bound: 843.2065410
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2066524, upper bound: 843.2068187
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -187.1232147, 636.2412109, -823.3643799, 823.3643799
1: -305.6914673, 777.5458984, -305.6914673, 777.5458984, -1083.2373047, 1083.2373047
2: -213.0952911, 822.5731201, -213.0952911, 822.5731201, -1035.6683350, 1035.6684570
3: -543.3413086, 792.1798706, -543.3413086, 792.1798706, -1335.5212402, 1335.5212402
4: -339.2044983, 844.9370728, -339.2044983, 844.9370728, -1184.1414795, 1184.1414795

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065814, upper bound: 843.2068505
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2066524, upper bound: 843.2071282
time: 0.67 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.26 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2086150, upper bound: 843.2091117
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2086150, upper bound: 843.2091117
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2086653, upper bound: 843.2091117
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2086653, upper bound: 843.2091117
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2096367, upper bound: 843.2096367
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2096367, upper bound: 843.2096870
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2096870, upper bound: 843.2096367
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2096870, upper bound: 843.2096870
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2085426, upper bound: 843.2074550
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2072649, upper bound: 843.2071630
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2085426, upper bound: 843.2077645
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2072649, upper bound: 843.2074725
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2088594, upper bound: 843.2079844
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2082051, upper bound: 843.2078292
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2088594, upper bound: 843.2082939
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2082052, upper bound: 843.2081387
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2071878, upper bound: 843.2070303
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2072649
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2071878, upper bound: 843.2079705
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2072649
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2074014, upper bound: 843.2074630
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2074724, upper bound: 843.2077407
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2074014, upper bound: 843.2082965
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2074725, upper bound: 843.2085883
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2063678, upper bound: 843.2061084
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2063430, upper bound: 843.2063430
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2063678, upper bound: 843.2061084
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2063430, upper bound: 843.2063430
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2065814, upper bound: 843.2065410
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2066524, upper bound: 843.2068187
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2065814, upper bound: 843.2068505
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.26
Output dim: 0, lower bound: -843.2066524, upper bound: 843.2071282

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -167.5513153, 569.7696533, -171.6107483, 583.5519409, -751.1032104, 741.3803711
1: -273.8763428, 696.5308838, -280.5059509, 713.3177490, -987.1939697, 977.0368652
2: -190.9233856, 737.1707153, -195.5651093, 754.9420776, -945.8654785, 932.7358398
3: -487.4682312, 709.7176514, -499.1630554, 726.9379272, -1214.4061279, 1208.8807373
4: -304.2389221, 757.0684814, -311.6126404, 775.2189941, -1079.4577637, 1068.6811523

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2096245
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2099345
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -204.7340698, 695.8717041, -171.6107483, 583.5519409, -788.2860107, 867.4824219
1: -337.6034241, 849.2514038, -280.5059509, 713.3177490, -1050.9211426, 1129.7573242
2: -233.1838074, 898.7626343, -195.5651093, 754.9420776, -988.1258545, 1094.3277588
3: -597.8302002, 865.3538818, -499.1630554, 726.9379272, -1324.7680664, 1364.5169678
4: -370.6482544, 923.5570679, -311.6126404, 775.2189941, -1145.8671875, 1235.1696777

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2096245
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2099345
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -167.5513153, 569.7696533, -208.0431061, 707.0399780, -874.5912476, 777.8126831
1: -273.8763428, 696.5308838, -342.9703064, 862.9215698, -1136.7978516, 1039.5009766
2: -190.9233856, 737.1707153, -237.0101166, 913.2547607, -1104.1779785, 974.1808472
3: -487.4682312, 709.7176514, -607.4349976, 879.5144653, -1366.9826660, 1317.1525879
4: -304.2389221, 757.0684814, -376.8020630, 938.5618896, -1242.8005371, 1133.8704834

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -204.7340698, 695.8717041, -208.0431061, 707.0399780, -911.7740479, 903.9147949
1: -337.6034241, 849.2514038, -342.9703064, 862.9215698, -1200.5250244, 1192.2214355
2: -233.1838074, 898.7626343, -237.0101166, 913.2547607, -1146.4384766, 1135.7727051
3: -597.8302002, 865.3538818, -607.4349976, 879.5144653, -1477.3446045, 1472.7888184
4: -370.6482544, 923.5570679, -376.8020630, 938.5618896, -1309.2099609, 1300.3590088

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2091117
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -168.3289490, 571.9808350, -171.6107483, 583.5519409, -751.8808594, 743.5915527
1: -274.5803833, 699.3826294, -280.5059509, 713.3177490, -987.8981323, 979.8885498
2: -191.8508453, 740.2653198, -195.5651093, 754.9420776, -946.7929077, 935.8304443
3: -489.2566528, 712.8433838, -499.1630554, 726.9379272, -1216.1944580, 1212.0062256
4: -305.7969666, 760.3158569, -311.6126404, 775.2189941, -1081.0158691, 1071.9283447

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091522, upper bound: 843.2101494
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091522, upper bound: 843.2104595
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -201.7331848, 684.9160767, -171.6107483, 583.5519409, -785.2851562, 856.5267944
1: -331.7389526, 836.4715576, -280.5059509, 713.3177490, -1045.0566406, 1116.9775391
2: -229.9456787, 885.3747559, -195.5651093, 754.9420776, -984.8877563, 1080.9398193
3: -588.3772583, 852.6589355, -499.1630554, 726.9379272, -1315.3151855, 1351.8217773
4: -365.7247314, 910.3051758, -311.6126404, 775.2189941, -1140.9437256, 1221.9178467

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091522, upper bound: 843.2101997
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091522, upper bound: 843.2105097
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -168.3289490, 571.9808350, -208.0431061, 707.0399780, -875.3688965, 780.0238647
1: -274.5803833, 699.3826294, -342.9703064, 862.9215698, -1137.5019531, 1042.3527832
2: -191.8508453, 740.2653198, -237.0101166, 913.2547607, -1105.1055908, 977.2754517
3: -489.2566528, 712.8433838, -607.4349976, 879.5144653, -1368.7708740, 1320.2781982
4: -305.7969666, 760.3158569, -376.8020630, 938.5618896, -1244.3585205, 1137.1175537

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2086150
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2096163
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -201.7331848, 684.9160767, -208.0431061, 707.0399780, -908.7731934, 892.9591064
1: -331.7389526, 836.4715576, -342.9703064, 862.9215698, -1194.6605225, 1179.4416504
2: -229.9456787, 885.3747559, -237.0101166, 913.2547607, -1143.2004395, 1122.3848877
3: -588.3772583, 852.6589355, -607.4349976, 879.5144653, -1467.8917236, 1460.0937500
4: -365.7247314, 910.3051758, -376.8020630, 938.5618896, -1304.2866211, 1287.1070557

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2086653
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2086653
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -172.2730255, 587.1818848, -188.4751282, 641.3044434, -813.5774536, 775.6569824
1: -280.8097839, 717.5310059, -308.6676941, 783.1837769, -1063.9935303, 1026.1984863
2: -196.2395630, 759.6019897, -214.4147797, 828.5700073, -1024.8095703, 974.0167236
3: -500.6431580, 730.5495605, -547.7684326, 797.9298096, -1298.5727539, 1278.3176270
4: -312.7213440, 779.9223633, -341.2081909, 850.9271851, -1163.6485596, 1121.1301270

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070150, upper bound: 843.2074024
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070303, upper bound: 843.2071630
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070303, upper bound: 843.2071630
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -179.2961731, 610.8264160, -188.4751282, 641.3044434, -820.6005859, 799.3015137
1: -292.4148254, 746.4031372, -308.6676941, 783.1837769, -1075.5986328, 1055.0705566
2: -204.1958008, 790.3706055, -214.4147797, 828.5700073, -1032.7657471, 1004.7853394
3: -521.0043945, 760.0877075, -547.7684326, 797.9298096, -1318.9339600, 1307.8557129
4: -325.3488159, 811.5093994, -341.2081909, 850.9271851, -1176.2760010, 1152.7174072

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2064714, upper bound: 843.2071630
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070303, upper bound: 843.2071630
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070303, upper bound: 843.2071630
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -172.2730255, 587.1818848, -187.1232147, 636.2412109, -808.5142212, 774.3049927
1: -280.8097839, 717.5310059, -305.6914673, 777.5458984, -1058.3555908, 1023.2224731
2: -196.2395630, 759.6019897, -213.0952911, 822.5731201, -1018.8126831, 972.6972656
3: -500.6431580, 730.5495605, -543.3413086, 792.1798706, -1292.8228760, 1273.8908691
4: -312.7213440, 779.9223633, -339.2044983, 844.9370728, -1157.6584473, 1119.1267090

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074996, upper bound: 843.2077119
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074630, upper bound: 843.2074014
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074630, upper bound: 843.2074724
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -179.2961731, 610.8264160, -187.1232147, 636.2412109, -815.5373535, 797.9495239
1: -292.4148254, 746.4031372, -305.6914673, 777.5458984, -1069.9606934, 1052.0943604
2: -204.1958008, 790.3706055, -213.0952911, 822.5731201, -1026.7687988, 1003.4658813
3: -521.0043945, 760.0877075, -543.3413086, 792.1798706, -1313.1840820, 1303.4289551
4: -325.3488159, 811.5093994, -339.2044983, 844.9370728, -1170.2858887, 1150.7138672

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069560, upper bound: 843.2074724
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074630, upper bound: 843.2074014
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074630, upper bound: 843.2074724
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -173.9681702, 592.3180542, -188.4751282, 641.3044434, -815.2725830, 780.7932129
1: -283.2098389, 723.9351807, -308.6676941, 783.1837769, -1066.3935547, 1032.6029053
2: -198.2450714, 766.4570923, -214.4147797, 828.5700073, -1026.8150635, 980.8718262
3: -505.2448425, 737.4182129, -547.7684326, 797.9298096, -1303.1746826, 1285.1862793
4: -315.9694824, 787.1150513, -341.2081909, 850.9271851, -1166.8967285, 1128.3228760

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080367, upper bound: 843.2079844
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079705, upper bound: 843.2078292
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079705, upper bound: 843.2078292
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -180.2173462, 613.6436768, -188.4751282, 641.3044434, -821.5217896, 802.1187744
1: -293.4556274, 749.9676514, -308.6676941, 783.1837769, -1076.6394043, 1058.6351318
2: -205.3023071, 794.1782837, -214.4147797, 828.5700073, -1033.8723145, 1008.5930786
3: -523.2786865, 763.9072266, -547.7684326, 797.9298096, -1321.2084961, 1311.6751709
4: -327.1666565, 815.4519653, -341.2081909, 850.9271851, -1178.0938721, 1156.6597900

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079711, upper bound: 843.2078292
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079705, upper bound: 843.2078292
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079705, upper bound: 843.2078292
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -173.9681702, 592.3180542, -187.1232147, 636.2412109, -810.2093506, 779.4412231
1: -283.2098389, 723.9351807, -305.6914673, 777.5458984, -1060.7557373, 1029.6265869
2: -198.2450714, 766.4570923, -213.0952911, 822.5731201, -1020.8181763, 979.5523682
3: -505.2448425, 737.4182129, -543.3413086, 792.1798706, -1297.4246826, 1280.7595215
4: -315.9694824, 787.1150513, -339.2044983, 844.9370728, -1160.9063721, 1126.3195801

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2085213, upper bound: 843.2082939
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2084032, upper bound: 843.2080676
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2084032, upper bound: 843.2081387
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -180.2173462, 613.6436768, -187.1232147, 636.2412109, -816.4585571, 800.7669067
1: -293.4556274, 749.9676514, -305.6914673, 777.5458984, -1071.0014648, 1055.6589355
2: -205.3023071, 794.1782837, -213.0952911, 822.5731201, -1027.8753662, 1007.2735596
3: -523.2786865, 763.9072266, -543.3413086, 792.1798706, -1315.4584961, 1307.2484131
4: -327.1666565, 815.4519653, -339.2044983, 844.9370728, -1172.1036377, 1154.6563721

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2084557, upper bound: 843.2081387
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2084032, upper bound: 843.2080676
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2084032, upper bound: 843.2081387
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -175.5612030, 598.3837891, -184.1930847, 626.9782715, -802.5394897, 782.5767212
1: -287.3129883, 730.6085205, -300.4424438, 766.2285767, -1053.5415039, 1031.0510254
2: -199.7297058, 772.8367310, -209.7961884, 811.2807007, -1011.0103760, 982.6327515
3: -510.0922241, 743.9570923, -535.2235718, 780.4590454, -1290.5512695, 1279.1806641
4: -317.8653870, 793.4354248, -334.2570190, 833.0983276, -1150.9637451, 1127.6923828

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2070303
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2070303
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -184.1430817, 626.8763428, -184.1930847, 626.9782715, -811.1213379, 811.0693359
1: -301.7005615, 765.4571533, -300.4424438, 766.2285767, -1067.9291992, 1065.8996582
2: -209.4716339, 809.8760986, -209.7961884, 811.2807007, -1020.7523193, 1019.6721191
3: -535.3128052, 779.7720947, -535.2235718, 780.4590454, -1315.7718506, 1314.9956055
4: -333.3037109, 831.6660767, -334.2570190, 833.0983276, -1166.4020996, 1165.9230957

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2072649
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2072649
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -175.5612030, 598.3837891, -185.4147034, 630.6639404, -806.2251587, 783.7984619
1: -287.3129883, 730.6085205, -302.0009155, 770.8518677, -1058.1647949, 1032.6093750
2: -199.7297058, 772.8367310, -211.2521362, 816.2529297, -1015.9826660, 984.0888672
3: -510.0922241, 743.9570923, -538.4077759, 785.4230957, -1295.5153809, 1282.3648682
4: -317.8653870, 793.4354248, -336.6280823, 838.2955933, -1156.1610107, 1130.0632324

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078292, upper bound: 843.2079705
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078292, upper bound: 843.2079705
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -184.1430817, 626.8763428, -185.4147034, 630.6639404, -814.8070068, 812.2910156
1: -301.7005615, 765.4571533, -302.0009155, 770.8518677, -1072.5524902, 1067.4580078
2: -209.4716339, 809.8760986, -211.2521362, 816.2529297, -1025.7244873, 1021.1281738
3: -535.3128052, 779.7720947, -538.4077759, 785.4230957, -1320.7358398, 1318.1796875
4: -333.3037109, 831.6660767, -336.6280823, 838.2955933, -1171.5993652, 1168.2940674

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078292, upper bound: 843.2082052
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078292, upper bound: 843.2082052
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -174.3998871, 593.9743042, -184.1930847, 626.9782715, -801.3781128, 778.1672363
1: -284.6991577, 725.7923584, -300.4424438, 766.2285767, -1050.9276123, 1026.2347412
2: -198.6237030, 767.6218262, -209.7961884, 811.2807007, -1009.9044189, 977.4179077
3: -506.3637085, 739.0524292, -535.2235718, 780.4590454, -1286.8227539, 1274.2760010
4: -316.1923523, 788.3237915, -334.2570190, 833.0983276, -1149.2906494, 1122.5808105

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074014, upper bound: 843.2074630
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074014, upper bound: 843.2074630
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -182.8314972, 621.9671631, -184.1930847, 626.9782715, -809.8097534, 806.1600342
1: -298.8067932, 759.9968262, -300.4424438, 766.2285767, -1065.0354004, 1060.4392090
2: -208.1809540, 804.0384521, -209.7961884, 811.2807007, -1019.4616089, 1013.8344727
3: -530.9179077, 774.1968384, -535.2235718, 780.4590454, -1311.3769531, 1309.4204102
4: -331.3768921, 825.8750000, -334.2570190, 833.0983276, -1164.4752197, 1160.1320801

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2077407
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074724, upper bound: 843.2077407
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -174.3998871, 593.9743042, -185.4147034, 630.6639404, -805.0637817, 779.3890381
1: -284.6991577, 725.7923584, -302.0009155, 770.8518677, -1055.5507812, 1027.7932129
2: -198.6237030, 767.6218262, -211.2521362, 816.2529297, -1014.8766479, 978.8739624
3: -506.3637085, 739.0524292, -538.4077759, 785.4230957, -1291.7868652, 1277.4599609
4: -316.1923523, 788.3237915, -336.6280823, 838.2955933, -1154.4879150, 1124.9515381

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2076413, upper bound: 843.2082965
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2076413, upper bound: 843.2082965
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -182.8314972, 621.9671631, -185.4147034, 630.6639404, -813.4954224, 807.3818359
1: -298.8067932, 759.9968262, -302.0009155, 770.8518677, -1069.6586914, 1061.9978027
2: -208.1809540, 804.0384521, -211.2521362, 816.2529297, -1024.4338379, 1015.2905273
3: -530.9179077, 774.1968384, -538.4077759, 785.4230957, -1316.3410645, 1312.6046143
4: -331.3768921, 825.8750000, -336.6280823, 838.2955933, -1169.6724854, 1162.5030518

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078378, upper bound: 843.2085883
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078378, upper bound: 843.2085883
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -175.5612030, 598.3837891, -188.4751282, 641.3044434, -816.8656616, 786.8588867
1: -287.3129883, 730.6085205, -308.6676941, 783.1837769, -1070.4968262, 1039.2760010
2: -199.7297058, 772.8367310, -214.4147797, 828.5700073, -1028.2996826, 987.2514038
3: -510.0922241, 743.9570923, -547.7684326, 797.9298096, -1308.0219727, 1291.7254639
4: -317.8653870, 793.4354248, -341.2081909, 850.9271851, -1168.7926025, 1134.6433105

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061084, upper bound: 843.2061084
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061084, upper bound: 843.2061084
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -184.1430817, 626.8763428, -188.4751282, 641.3044434, -825.4475098, 815.3514404
1: -301.7005615, 765.4571533, -308.6676941, 783.1837769, -1084.8842773, 1074.1248779
2: -209.4716339, 809.8760986, -214.4147797, 828.5700073, -1038.0416260, 1024.2908936
3: -535.3128052, 779.7720947, -547.7684326, 797.9298096, -1333.2425537, 1327.5402832
4: -333.3037109, 831.6660767, -341.2081909, 850.9271851, -1184.2309570, 1172.8741455

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061084, upper bound: 843.2063430
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061084, upper bound: 843.2063430
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -175.5612030, 598.3837891, -187.1232147, 636.2412109, -811.8024292, 785.5069580
1: -287.3129883, 730.6085205, -305.6914673, 777.5458984, -1064.8588867, 1036.2999268
2: -199.7297058, 772.8367310, -213.0952911, 822.5731201, -1022.3028564, 985.9320068
3: -510.0922241, 743.9570923, -543.3413086, 792.1798706, -1302.2720947, 1287.2983398
4: -317.8653870, 793.4354248, -339.2044983, 844.9370728, -1162.8024902, 1132.6398926

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065410, upper bound: 843.2063468
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065410, upper bound: 843.2064178
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -184.1430817, 626.8763428, -187.1232147, 636.2412109, -820.3842773, 813.9995728
1: -301.7005615, 765.4571533, -305.6914673, 777.5458984, -1079.2464600, 1071.1486816
2: -209.4716339, 809.8760986, -213.0952911, 822.5731201, -1032.0447998, 1022.9713745
3: -535.3128052, 779.7720947, -543.3413086, 792.1798706, -1327.4926758, 1323.1134033
4: -333.3037109, 831.6660767, -339.2044983, 844.9370728, -1178.2407227, 1170.8706055

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065410, upper bound: 843.2065814
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065410, upper bound: 843.2066524
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -174.3998871, 593.9743042, -188.4751282, 641.3044434, -815.7042847, 782.4494629
1: -284.6991577, 725.7923584, -308.6676941, 783.1837769, -1067.8829346, 1034.4598389
2: -198.6237030, 767.6218262, -214.4147797, 828.5700073, -1027.1936035, 982.0366211
3: -506.3637085, 739.0524292, -547.7684326, 797.9298096, -1304.2933350, 1286.8204346
4: -316.1923523, 788.3237915, -341.2081909, 850.9271851, -1167.1195068, 1129.5316162

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063468, upper bound: 843.2065410
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063468, upper bound: 843.2065410
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -182.8314972, 621.9671631, -188.4751282, 641.3044434, -824.1359253, 810.4422607
1: -298.8067932, 759.9968262, -308.6676941, 783.1837769, -1081.9906006, 1068.6645508
2: -208.1809540, 804.0384521, -214.4147797, 828.5700073, -1036.7509766, 1018.4531860
3: -530.9179077, 774.1968384, -547.7684326, 797.9298096, -1328.8476562, 1321.9650879
4: -331.3768921, 825.8750000, -341.2081909, 850.9271851, -1182.3040771, 1167.0831299

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2064178, upper bound: 843.2068187
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2064178, upper bound: 843.2068187
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -174.3998871, 593.9743042, -187.1232147, 636.2412109, -810.6410522, 781.0974731
1: -284.6991577, 725.7923584, -305.6914673, 777.5458984, -1062.2449951, 1031.4836426
2: -198.6237030, 767.6218262, -213.0952911, 822.5731201, -1021.1968384, 980.7171021
3: -506.3637085, 739.0524292, -543.3413086, 792.1798706, -1298.5434570, 1282.3937988
4: -316.1923523, 788.3237915, -339.2044983, 844.9370728, -1161.1292725, 1127.5281982

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2066231, upper bound: 843.2067794
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2066231, upper bound: 843.2068505
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -182.8314972, 621.9671631, -187.1232147, 636.2412109, -819.0726929, 809.0902710
1: -298.8067932, 759.9968262, -305.6914673, 777.5458984, -1076.3526611, 1065.6882324
2: -208.1809540, 804.0384521, -213.0952911, 822.5731201, -1030.7540283, 1017.1337280
3: -530.9179077, 774.1968384, -543.3413086, 792.1798706, -1323.0977783, 1317.5380859
4: -331.3768921, 825.8750000, -339.2044983, 844.9370728, -1176.3138428, 1165.0794678

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067781, upper bound: 843.2070571
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067781, upper bound: 843.2071282
time: 0.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.29 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2096245
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2099345
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2096245
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2099345
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2091117
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2091522, upper bound: 843.2101494
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2091522, upper bound: 843.2104595
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2091522, upper bound: 843.2101997
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2091522, upper bound: 843.2105097
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2086150
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2096163
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2086653
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2086653
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2070303, upper bound: 843.2071630
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2070303, upper bound: 843.2071630
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2070303, upper bound: 843.2071630
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2070303, upper bound: 843.2071630
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2074630, upper bound: 843.2074014
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2074630, upper bound: 843.2074724
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2074630, upper bound: 843.2074014
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2074630, upper bound: 843.2074724
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2079705, upper bound: 843.2078292
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2079705, upper bound: 843.2078292
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2079705, upper bound: 843.2078292
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2079705, upper bound: 843.2078292
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2084032, upper bound: 843.2080676
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2084032, upper bound: 843.2081387
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2084032, upper bound: 843.2080676
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2084032, upper bound: 843.2081387
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2070303
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2070303
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2072649
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2072649
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2078292, upper bound: 843.2079705
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2078292, upper bound: 843.2079705
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2078292, upper bound: 843.2082052
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2078292, upper bound: 843.2082052
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2074014, upper bound: 843.2074630
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2074014, upper bound: 843.2074630
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2077407
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2074724, upper bound: 843.2077407
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2076413, upper bound: 843.2082965
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2076413, upper bound: 843.2082965
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2078378, upper bound: 843.2085883
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2078378, upper bound: 843.2085883
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2061084, upper bound: 843.2061084
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2061084, upper bound: 843.2061084
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2061084, upper bound: 843.2063430
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2061084, upper bound: 843.2063430
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2065410, upper bound: 843.2063468
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2065410, upper bound: 843.2064178
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2065410, upper bound: 843.2065814
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2065410, upper bound: 843.2066524
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2063468, upper bound: 843.2065410
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2063468, upper bound: 843.2065410
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2064178, upper bound: 843.2068187
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2064178, upper bound: 843.2068187
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2066231, upper bound: 843.2067794
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2066231, upper bound: 843.2068505
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2067781, upper bound: 843.2070571
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 0, lower bound: -843.2067781, upper bound: 843.2071282

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -167.5513153, 569.7696533, -167.5513153, 569.7696533, -737.3209229, 737.3209229
1: -273.8763428, 696.5308838, -273.8763428, 696.5308838, -970.4071655, 970.4071655
2: -190.9233856, 737.1707153, -190.9233856, 737.1707153, -928.0941162, 928.0941162
3: -487.4682312, 709.7176514, -487.4682312, 709.7176514, -1197.1859131, 1197.1859131
4: -304.2389221, 757.0684814, -304.2389221, 757.0684814, -1061.3073730, 1061.3073730

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2092815, upper bound: 843.2083712
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079882, upper bound: 843.2079882
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -167.5513153, 569.7696533, -168.3289490, 571.9808350, -739.5321045, 738.0985718
1: -273.8763428, 696.5308838, -274.5803833, 699.3826294, -973.2587891, 971.1112671
2: -190.9233856, 737.1707153, -191.8508453, 740.2653198, -931.1887207, 929.0214844
3: -487.4682312, 709.7176514, -489.2566528, 712.8433838, -1200.3115234, 1198.9742432
4: -304.2389221, 757.0684814, -305.7969666, 760.3158569, -1064.5545654, 1062.8654785

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2092815, upper bound: 843.2093114
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079882, upper bound: 843.2079882
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -204.7340698, 695.8717041, -167.5513153, 569.7696533, -774.5037231, 863.4230347
1: -337.6034241, 849.2514038, -273.8763428, 696.5308838, -1034.1342773, 1123.1276855
2: -233.1838074, 898.7626343, -190.9233856, 737.1707153, -970.3544922, 1089.6859131
3: -597.8302002, 865.3538818, -487.4682312, 709.7176514, -1307.5478516, 1352.8221436
4: -370.6482544, 923.5570679, -304.2389221, 757.0684814, -1127.7167969, 1227.7958984

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2083244
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2080850
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -204.7340698, 695.8717041, -168.3289490, 571.9808350, -776.7149048, 864.2006836
1: -337.6034241, 849.2514038, -274.5803833, 699.3826294, -1036.9860840, 1123.8317871
2: -233.1838074, 898.7626343, -191.8508453, 740.2653198, -973.4490356, 1090.6135254
3: -597.8302002, 865.3538818, -489.2566528, 712.8433838, -1310.6733398, 1354.6104736
4: -370.6482544, 923.5570679, -305.7969666, 760.3158569, -1130.9639893, 1229.3540039

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2092646
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2090252
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -167.5513153, 569.7696533, -204.7340698, 695.8717041, -863.4230347, 774.5037231
1: -273.8763428, 696.5308838, -337.6034241, 849.2514038, -1123.1276855, 1034.1342773
2: -190.9233856, 737.1707153, -233.1838074, 898.7626343, -1089.6860352, 970.3544922
3: -487.4682312, 709.7176514, -597.8302002, 865.3538818, -1352.8221436, 1307.5478516
4: -304.2389221, 757.0684814, -370.6482544, 923.5570679, -1227.7958984, 1127.7167969

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2093553, upper bound: 843.2075865
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080850, upper bound: 843.2072034
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -167.5513153, 569.7696533, -202.6466675, 687.9989624, -855.5502319, 772.4163208
1: -273.8763428, 696.5308838, -333.1848145, 840.2247314, -1114.1009521, 1029.7156982
2: -190.9233856, 737.1707153, -230.9812775, 889.4022217, -1080.3255615, 968.1519775
3: -487.4682312, 709.7176514, -591.0267944, 856.4942627, -1343.9624023, 1300.7443848
4: -304.2389221, 757.0684814, -367.3989258, 914.4356689, -1218.6745605, 1124.4674072

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2093553, upper bound: 843.2090862
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080850, upper bound: 843.2087032
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -204.7340698, 695.8717041, -204.7340698, 695.8717041, -900.6057739, 900.6057739
1: -337.6034241, 849.2514038, -337.6034241, 849.2514038, -1186.8547363, 1186.8547363
2: -233.1838074, 898.7626343, -233.1838074, 898.7626343, -1131.9464111, 1131.9464111
3: -597.8302002, 865.3538818, -597.8302002, 865.3538818, -1463.1840820, 1463.1840820
4: -370.6482544, 923.5570679, -370.6482544, 923.5570679, -1294.2053223, 1294.2053223

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2075397
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2073002
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -204.7340698, 695.8717041, -202.6466675, 687.9989624, -892.7330322, 898.5183716
1: -337.6034241, 849.2514038, -333.1848145, 840.2247314, -1177.8278809, 1182.4360352
2: -233.1838074, 898.7626343, -230.9812775, 889.4022217, -1122.5860596, 1129.7437744
3: -597.8302002, 865.3538818, -591.0267944, 856.4942627, -1454.3242188, 1456.3806152
4: -370.6482544, 923.5570679, -367.3989258, 914.4356689, -1285.0839844, 1290.9560547

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2090395
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2088000
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -168.3289490, 571.9808350, -167.5513153, 569.7696533, -738.0985718, 739.5321045
1: -274.5803833, 699.3826294, -273.8763428, 696.5308838, -971.1112671, 973.2587891
2: -191.8508453, 740.2653198, -190.9233856, 737.1707153, -929.0214844, 931.1887207
3: -489.2566528, 712.8433838, -487.4682312, 709.7176514, -1198.9742432, 1200.3115234
4: -305.7969666, 760.3158569, -304.2389221, 757.0684814, -1062.8654785, 1064.5546875

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2095915, upper bound: 843.2088535
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2089284, upper bound: 843.2086498
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -168.3289490, 571.9808350, -168.3289490, 571.9808350, -740.3097534, 740.3097534
1: -274.5803833, 699.3826294, -274.5803833, 699.3826294, -973.9630127, 973.9630127
2: -191.8508453, 740.2653198, -191.8508453, 740.2653198, -932.1160889, 932.1160278
3: -489.2566528, 712.8433838, -489.2566528, 712.8433838, -1202.0997314, 1202.0997314
4: -305.7969666, 760.3158569, -305.7969666, 760.3158569, -1066.1127930, 1066.1127930

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2095915, upper bound: 843.2088535
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2089284, upper bound: 843.2095901
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -201.7331848, 684.9160767, -167.5513153, 569.7696533, -771.5028076, 852.4674072
1: -331.7389526, 836.4715576, -273.8763428, 696.5308838, -1028.2697754, 1110.3479004
2: -229.9456787, 885.3747559, -190.9233856, 737.1707153, -967.1163940, 1076.2980957
3: -588.3772583, 852.6589355, -487.4682312, 709.7176514, -1298.0949707, 1340.1270752
4: -365.7247314, 910.3051758, -304.2389221, 757.0684814, -1122.7932129, 1214.5440674

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2089064
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2087512
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -201.7331848, 684.9160767, -168.3289490, 571.9808350, -773.7139893, 853.2449951
1: -331.7389526, 836.4715576, -274.5803833, 699.3826294, -1031.1215820, 1111.0520020
2: -229.9456787, 885.3747559, -191.8508453, 740.2653198, -970.2109985, 1077.2255859
3: -588.3772583, 852.6589355, -489.2566528, 712.8433838, -1301.2205811, 1341.9152832
4: -365.7247314, 910.3051758, -305.7969666, 760.3158569, -1126.0405273, 1216.1020508

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2098466
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2087512
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -168.3289490, 571.9808350, -204.7340698, 695.8717041, -864.2006836, 776.7149048
1: -274.5803833, 699.3826294, -337.6034241, 849.2514038, -1123.8317871, 1036.9859619
2: -191.8508453, 740.2653198, -233.1838074, 898.7626343, -1090.6135254, 973.4490967
3: -489.2566528, 712.8433838, -597.8302002, 865.3538818, -1354.6104736, 1310.6733398
4: -305.7969666, 760.3158569, -370.6482544, 923.5570679, -1229.3540039, 1130.9639893

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096816, upper bound: 843.2080687
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2090252, upper bound: 843.2078651
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -168.3289490, 571.9808350, -202.6466675, 687.9989624, -856.3278809, 774.6275024
1: -274.5803833, 699.3826294, -333.1848145, 840.2247314, -1114.8050537, 1032.5673828
2: -191.8508453, 740.2653198, -230.9812775, 889.4022217, -1081.2530518, 971.2465820
3: -489.2566528, 712.8433838, -591.0267944, 856.4942627, -1345.7504883, 1303.8698730
4: -305.7969666, 760.3158569, -367.3989258, 914.4356689, -1220.2326660, 1127.7147217

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096817, upper bound: 843.2094802
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2090252, upper bound: 843.2093649
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -201.7331848, 684.9160767, -204.7340698, 695.8717041, -897.6048584, 889.6501465
1: -331.7389526, 836.4715576, -337.6034241, 849.2514038, -1180.9902344, 1174.0749512
2: -229.9456787, 885.3747559, -233.1838074, 898.7626343, -1128.7082520, 1118.5585938
3: -588.3772583, 852.6589355, -597.8302002, 865.3538818, -1453.7312012, 1450.4890137
4: -365.7247314, 910.3051758, -370.6482544, 923.5570679, -1289.2817383, 1280.9533691

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2081217
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2079665
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -201.7331848, 684.9160767, -202.6466675, 687.9989624, -889.7321777, 887.5627441
1: -331.7389526, 836.4715576, -333.1848145, 840.2247314, -1171.9636230, 1169.6562500
2: -229.9456787, 885.3747559, -230.9812775, 889.4022217, -1119.3479004, 1116.3559570
3: -588.3772583, 852.6589355, -591.0267944, 856.4942627, -1444.8715820, 1443.6854248
4: -365.7247314, 910.3051758, -367.3989258, 914.4356689, -1280.1604004, 1277.7041016

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2095895
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2094663
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -172.2730255, 587.1818848, -175.5612030, 598.3837891, -770.6567993, 762.7430420
1: -280.8097839, 717.5310059, -287.3129883, 730.6085205, -1011.4182739, 1004.8439331
2: -196.2395630, 759.6019897, -199.7297058, 772.8367310, -969.0762939, 959.3316650
3: -500.6431580, 730.5495605, -510.0922241, 743.9570923, -1244.6002197, 1240.6418457
4: -312.7213440, 779.9223633, -317.8653870, 793.4354248, -1106.1567383, 1097.7877197

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063976, upper bound: 843.2064441
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079774, upper bound: 843.2064441
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -172.2730255, 587.1818848, -184.1430817, 626.8763428, -799.1493530, 771.3249512
1: -280.8097839, 717.5310059, -301.7005615, 765.4571533, -1046.2668457, 1019.2315063
2: -196.2395630, 759.6019897, -209.4716339, 809.8760986, -1006.1156616, 969.0736084
3: -500.6431580, 730.5495605, -535.3128052, 779.7720947, -1280.4151611, 1265.8623047
4: -312.7213440, 779.9223633, -333.3037109, 831.6660767, -1144.3874512, 1113.2259521

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063976, upper bound: 843.2064441
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079774, upper bound: 843.2067206
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -179.2961731, 610.8264160, -175.5612030, 598.3837891, -777.6799316, 786.3875732
1: -292.4148254, 746.4031372, -287.3129883, 730.6085205, -1023.0233154, 1033.7160645
2: -204.1958008, 790.3706055, -199.7297058, 772.8367310, -977.0325317, 990.1003418
3: -521.0043945, 760.0877075, -510.0922241, 743.9570923, -1264.9614258, 1270.1799316
4: -325.3488159, 811.5093994, -317.8653870, 793.4354248, -1118.7841797, 1129.3747559

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060500, upper bound: 843.2063248
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2066041, upper bound: 843.2060465
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -179.2961731, 610.8264160, -184.1430817, 626.8763428, -806.1724854, 794.9694824
1: -292.4148254, 746.4031372, -301.7005615, 765.4571533, -1057.8719482, 1048.1037598
2: -204.1958008, 790.3706055, -209.4716339, 809.8760986, -1014.0718994, 999.8422241
3: -521.0043945, 760.0877075, -535.3128052, 779.7720947, -1300.7763672, 1295.4003906
4: -325.3488159, 811.5093994, -333.3037109, 831.6660767, -1157.0148926, 1144.8131104

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060500, upper bound: 843.2063248
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2066041, upper bound: 843.2060465
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -172.2730255, 587.1818848, -174.3998871, 593.9743042, -766.2473145, 761.5816650
1: -280.8097839, 717.5310059, -284.6991577, 725.7923584, -1006.6021729, 1002.2301636
2: -196.2395630, 759.6019897, -198.6237030, 767.6218262, -963.8613892, 958.2257080
3: -500.6431580, 730.5495605, -506.3637085, 739.0524292, -1239.6953125, 1236.9130859
4: -312.7213440, 779.9223633, -316.1923523, 788.3237915, -1101.0451660, 1096.1145020

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2066194, upper bound: 843.2064999
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081992, upper bound: 843.2067765
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -172.2730255, 587.1818848, -182.8314972, 621.9671631, -794.2401123, 770.0133057
1: -280.8097839, 717.5310059, -298.8067932, 759.9968262, -1040.8066406, 1016.3377075
2: -196.2395630, 759.6019897, -208.1809540, 804.0384521, -1000.2780151, 967.7828979
3: -500.6431580, 730.5495605, -530.9179077, 774.1968384, -1274.8399658, 1261.4674072
4: -312.7213440, 779.9223633, -331.3768921, 825.8750000, -1138.5963135, 1111.2990723

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2066194, upper bound: 843.2064999
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081992, upper bound: 843.2067765
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -179.2961731, 610.8264160, -174.3998871, 593.9743042, -773.2704468, 785.2261963
1: -292.4148254, 746.4031372, -284.6991577, 725.7923584, -1018.2071533, 1031.1020508
2: -204.1958008, 790.3706055, -198.6237030, 767.6218262, -971.8176270, 988.9943237
3: -521.0043945, 760.0877075, -506.3637085, 739.0524292, -1260.0566406, 1266.4511719
4: -325.3488159, 811.5093994, -316.1923523, 788.3237915, -1113.6726074, 1127.7017822

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2062718, upper bound: 843.2063806
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2068259, upper bound: 843.2061023
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -179.2961731, 610.8264160, -182.8314972, 621.9671631, -801.2632446, 793.6578369
1: -292.4148254, 746.4031372, -298.8067932, 759.9968262, -1052.4116211, 1045.2098389
2: -204.1958008, 790.3706055, -208.1809540, 804.0384521, -1008.2342529, 998.5514526
3: -521.0043945, 760.0877075, -530.9179077, 774.1968384, -1295.2011719, 1291.0054932
4: -325.3488159, 811.5093994, -331.3768921, 825.8750000, -1151.2238770, 1142.8862305

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2062718, upper bound: 843.2063806
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2068259, upper bound: 843.2061023
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -173.9681702, 592.3180542, -175.5612030, 598.3837891, -772.3519287, 767.8792725
1: -283.2098389, 723.9351807, -287.3129883, 730.6085205, -1013.8182983, 1011.2481079
2: -198.2450714, 766.4570923, -199.7297058, 772.8367310, -971.0817871, 966.1867676
3: -505.2448425, 737.4182129, -510.0922241, 743.9570923, -1249.2019043, 1247.5104980
4: -315.9694824, 787.1150513, -317.8653870, 793.4354248, -1109.4049072, 1104.9804688

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2073725, upper bound: 843.2070315
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2083144, upper bound: 843.2072258
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -173.9681702, 592.3180542, -184.1430817, 626.8763428, -800.8444824, 776.4611206
1: -283.2098389, 723.9351807, -301.7005615, 765.4571533, -1048.6669922, 1025.6357422
2: -198.2450714, 766.4570923, -209.4716339, 809.8760986, -1008.1211548, 975.9287109
3: -505.2448425, 737.4182129, -535.3128052, 779.7720947, -1285.0169678, 1272.7309570
4: -315.9694824, 787.1150513, -333.3037109, 831.6660767, -1147.6354980, 1120.4187012

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2073725, upper bound: 843.2070315
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2083144, upper bound: 843.2072258
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -180.2173462, 613.6436768, -175.5612030, 598.3837891, -778.6011353, 789.2048950
1: -293.4556274, 749.9676514, -287.3129883, 730.6085205, -1024.0642090, 1037.2806396
2: -205.3023071, 794.1782837, -199.7297058, 772.8367310, -978.1389160, 993.9079590
3: -523.2786865, 763.9072266, -510.0922241, 743.9570923, -1267.2358398, 1273.9993896
4: -327.1666565, 815.4519653, -317.8653870, 793.4354248, -1120.6020508, 1133.3173828

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071645, upper bound: 843.2069760
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070668, upper bound: 843.2065856
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -180.2173462, 613.6436768, -184.1430817, 626.8763428, -807.0936890, 797.7867432
1: -293.4556274, 749.9676514, -301.7005615, 765.4571533, -1058.9128418, 1051.6682129
2: -205.3023071, 794.1782837, -209.4716339, 809.8760986, -1015.1783447, 1003.6499023
3: -523.2786865, 763.9072266, -535.3128052, 779.7720947, -1303.0507812, 1299.2199707
4: -327.1666565, 815.4519653, -333.3037109, 831.6660767, -1158.8327637, 1148.7556152

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071645, upper bound: 843.2069760
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070668, upper bound: 843.2065856
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -173.9681702, 592.3180542, -174.3998871, 593.9743042, -767.9425049, 766.7178955
1: -283.2098389, 723.9351807, -284.6991577, 725.7923584, -1009.0021362, 1008.6343384
2: -198.2450714, 766.4570923, -198.6237030, 767.6218262, -965.8668823, 965.0808105
3: -505.2448425, 737.4182129, -506.3637085, 739.0524292, -1244.2972412, 1243.7817383
4: -315.9694824, 787.1150513, -316.1923523, 788.3237915, -1104.2932129, 1103.3073730

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2075943, upper bound: 843.2070873
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2085362, upper bound: 843.2072817
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -173.9681702, 592.3180542, -182.8314972, 621.9671631, -795.9353027, 775.1495361
1: -283.2098389, 723.9351807, -298.8067932, 759.9968262, -1043.2066650, 1022.7419434
2: -198.2450714, 766.4570923, -208.1809540, 804.0384521, -1002.2835083, 974.6380005
3: -505.2448425, 737.4182129, -530.9179077, 774.1968384, -1279.4416504, 1268.3360596
4: -315.9694824, 787.1150513, -331.3768921, 825.8750000, -1141.8444824, 1118.4919434

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2075943, upper bound: 843.2070873
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2085362, upper bound: 843.2072817
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -180.2173462, 613.6436768, -174.3998871, 593.9743042, -774.1916504, 788.0435791
1: -293.4556274, 749.9676514, -284.6991577, 725.7923584, -1019.2479858, 1034.6665039
2: -205.3023071, 794.1782837, -198.6237030, 767.6218262, -972.9241333, 992.8020020
3: -523.2786865, 763.9072266, -506.3637085, 739.0524292, -1262.3310547, 1270.2706299
4: -327.1666565, 815.4519653, -316.1923523, 788.3237915, -1115.4904785, 1131.6441650

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2073863, upper bound: 843.2070319
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072886, upper bound: 843.2066415
time: 1.08 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.12 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2092815, upper bound: 843.2083712
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2079882, upper bound: 843.2079882
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2092815, upper bound: 843.2093114
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2079882, upper bound: 843.2079882
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2083244
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2080850
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2092646
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2090252
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2093553, upper bound: 843.2075865
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2080850, upper bound: 843.2072034
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2093553, upper bound: 843.2090862
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2080850, upper bound: 843.2087032
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2075397
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2073002
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2090395
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2088000
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2095915, upper bound: 843.2088535
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2089284, upper bound: 843.2086498
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2095915, upper bound: 843.2088535
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2089284, upper bound: 843.2095901
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2089064
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2087512
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2098466
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2087512
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2096816, upper bound: 843.2080687
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2090252, upper bound: 843.2078651
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2096817, upper bound: 843.2094802
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2090252, upper bound: 843.2093649
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2081217
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2079665
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2095895
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2094663
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2063976, upper bound: 843.2064441
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2079774, upper bound: 843.2064441
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2063976, upper bound: 843.2064441
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2079774, upper bound: 843.2067206
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2060500, upper bound: 843.2063248
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2066041, upper bound: 843.2060465
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2060500, upper bound: 843.2063248
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2066041, upper bound: 843.2060465
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2066194, upper bound: 843.2064999
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2081992, upper bound: 843.2067765
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2066194, upper bound: 843.2064999
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2081992, upper bound: 843.2067765
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2062718, upper bound: 843.2063806
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2068259, upper bound: 843.2061023
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2062718, upper bound: 843.2063806
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2068259, upper bound: 843.2061023
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2073725, upper bound: 843.2070315
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2083144, upper bound: 843.2072258
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2073725, upper bound: 843.2070315
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2083144, upper bound: 843.2072258
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2071645, upper bound: 843.2069760
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2070668, upper bound: 843.2065856
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2071645, upper bound: 843.2069760
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2070668, upper bound: 843.2065856
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2075943, upper bound: 843.2070873
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2085362, upper bound: 843.2072817
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2075943, upper bound: 843.2070873
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2085362, upper bound: 843.2072817
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2073863, upper bound: 843.2070319
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.12
Output dim: 0, lower bound: -843.2072886, upper bound: 843.2066415
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2084032, upper bound: 843.2081387
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2070303
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2070303
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2072649
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2072649
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2078292, upper bound: 843.2079705
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2078292, upper bound: 843.2079705
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2078292, upper bound: 843.2082052
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2078292, upper bound: 843.2082052
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2074014, upper bound: 843.2074630
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2074014, upper bound: 843.2074630
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2071630, upper bound: 843.2077407
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2074724, upper bound: 843.2077407
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2076413, upper bound: 843.2082965
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2076413, upper bound: 843.2082965
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2078378, upper bound: 843.2085883
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2078378, upper bound: 843.2085883
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2061084, upper bound: 843.2061084
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2061084, upper bound: 843.2061084
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2061084, upper bound: 843.2063430
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2061084, upper bound: 843.2063430
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2065410, upper bound: 843.2063468
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2065410, upper bound: 843.2064178
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2065410, upper bound: 843.2065814
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2065410, upper bound: 843.2066524
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2063468, upper bound: 843.2065410
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2063468, upper bound: 843.2065410
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2064178, upper bound: 843.2068187
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2064178, upper bound: 843.2068187
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2066231, upper bound: 843.2067794
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2066231, upper bound: 843.2068505
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2067781, upper bound: 843.2070571
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.12
Output dim: 0, lower bound: -843.2067781, upper bound: 843.2071282
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=901.5203857421875
rel_dist={0: [-843.2117555388025, 843.2117555388029]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2095946, upper bound: 843.2088391
time: 0.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077508, upper bound: 843.2077508
time: 0.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.58 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -843.2095946, upper bound: 843.2088391
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -843.2077508, upper bound: 843.2077508

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -188.3924713, 641.2564087, -202.3066254, 688.7346802, -877.1271362, 843.5630493
1: -307.3268738, 783.6063843, -330.6458130, 841.2741699, -1148.6010742, 1114.2521973
2: -214.6040497, 829.6708374, -230.3630981, 890.6596680, -1105.2636719, 1060.0339355
3: -547.3501587, 798.2861328, -587.6779175, 856.9144287, -1404.2646484, 1385.9639893
4: -341.8905029, 851.8900757, -366.6545715, 914.2129517, -1256.1031494, 1218.5446777

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2089426, upper bound: 843.2088307
time: 0.95 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077508, upper bound: 843.2077508
time: 0.67 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077508, upper bound: 843.2077508
time: 0.69 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -191.9384003, 653.1663818, -203.1746063, 691.5394897, -883.4779053, 856.3409424
1: -314.3370667, 797.7437744, -332.3217468, 844.6906128, -1159.0274658, 1130.0653076
2: -218.4419098, 843.8818970, -231.3206787, 894.1173096, -1112.5592041, 1075.2026367
3: -557.8200684, 812.7958374, -590.2221680, 860.4089966, -1418.2288818, 1403.0179443
4: -347.6372986, 866.6468506, -368.1190186, 917.7901611, -1265.4273682, 1234.7658691

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077508, upper bound: 843.2077508
time: 0.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077508, upper bound: 843.2077508
time: 0.86 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.73 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 0, lower bound: -843.2077508, upper bound: 843.2077508
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 0, lower bound: -843.2077508, upper bound: 843.2077508
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 0, lower bound: -843.2077508, upper bound: 843.2077508
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 0, lower bound: -843.2077508, upper bound: 843.2077508

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -188.3924713, 641.2564087, -188.3924713, 641.2564087, -829.6488647, 829.6488647
1: -307.3268738, 783.6063843, -307.3268738, 783.6063843, -1090.9332275, 1090.9332275
2: -214.6040497, 829.6708374, -214.6040497, 829.6708374, -1044.2749023, 1044.2749023
3: -547.3501587, 798.2861328, -547.3501587, 798.2861328, -1345.6362305, 1345.6362305
4: -341.8905029, 851.8900757, -341.8905029, 851.8900757, -1193.7805176, 1193.7805176

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2085432, upper bound: 843.2078159
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091336, upper bound: 843.2083269
time: 0.95 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -188.3924713, 641.2564087, -191.9384003, 653.1663818, -841.5588379, 833.1948242
1: -307.3268738, 783.6063843, -314.3370667, 797.7437744, -1105.0705566, 1097.9434814
2: -214.6040497, 829.6708374, -218.4419098, 843.8818970, -1058.4859619, 1048.1127930
3: -547.3501587, 798.2861328, -557.8200684, 812.7958374, -1360.1459961, 1356.1062012
4: -341.8905029, 851.8900757, -347.6372986, 866.6468506, -1208.5373535, 1199.5273438

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2085432, upper bound: 843.2078159
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091336, upper bound: 843.2083269
time: 0.81 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -191.9384003, 653.1663818, -188.3924713, 641.2564087, -833.1948242, 841.5588379
1: -314.3370667, 797.7437744, -307.3268738, 783.6063843, -1097.9434814, 1105.0705566
2: -218.4419098, 843.8818970, -214.6040497, 829.6708374, -1048.1127930, 1058.4859619
3: -557.8200684, 812.7958374, -547.3501587, 798.2861328, -1356.1062012, 1360.1459961
4: -347.6372986, 866.6468506, -341.8905029, 851.8900757, -1199.5273438, 1208.5373535

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067933, upper bound: 843.2067038
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071776, upper bound: 843.2071776
time: 0.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -191.9384003, 653.1663818, -191.9384003, 653.1663818, -845.1047974, 845.1047974
1: -314.3370667, 797.7437744, -314.3370667, 797.7437744, -1112.0806885, 1112.0806885
2: -218.4419098, 843.8818970, -218.4419098, 843.8818970, -1062.3238525, 1062.3238525
3: -557.8200684, 812.7958374, -557.8200684, 812.7958374, -1370.6158447, 1370.6158447
4: -347.6372986, 866.6468506, -347.6372986, 866.6468506, -1214.2841797, 1214.2841797

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067933, upper bound: 843.2067038
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071776, upper bound: 843.2071776
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.52 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -843.2085432, upper bound: 843.2078159
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -843.2091336, upper bound: 843.2083269
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -843.2085432, upper bound: 843.2078159
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -843.2091336, upper bound: 843.2083269
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -843.2067933, upper bound: 843.2067038
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -843.2071776, upper bound: 843.2071776
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -843.2067933, upper bound: 843.2067038
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -843.2071776, upper bound: 843.2071776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -184.1930847, 626.9782715, -188.3924713, 641.2564087, -825.4493408, 815.3707275
1: -300.4424438, 766.2285767, -307.3268738, 783.6063843, -1084.0488281, 1073.5554199
2: -209.7961884, 811.2807007, -214.6040497, 829.6708374, -1039.4670410, 1025.8847656
3: -535.2235718, 780.4590454, -547.3501587, 798.2861328, -1333.5097656, 1327.8092041
4: -334.2570190, 833.0983276, -341.8905029, 851.8900757, -1186.1470947, 1174.9887695

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2086150, upper bound: 843.2099345
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2086653, upper bound: 843.2091117
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -185.4147034, 630.6639404, -188.0576477, 640.1040649, -825.5187988, 818.7215576
1: -302.0009155, 770.8518677, -306.7427063, 782.2080688, -1084.2088623, 1077.5944824
2: -211.2521362, 816.2529297, -214.2190247, 828.1894531, -1039.4415283, 1030.4718018
3: -538.4077759, 785.4230957, -546.3464966, 796.8613892, -1335.2691650, 1331.7695312
4: -336.6280823, 838.2955933, -341.2821960, 850.3778076, -1187.0053711, 1179.5777588

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096367, upper bound: 843.2105097
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096870, upper bound: 843.2096870
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -184.1930847, 626.9782715, -191.9384003, 653.1663818, -837.3593140, 818.9166870
1: -300.4424438, 766.2285767, -314.3370667, 797.7437744, -1098.1860352, 1080.5656738
2: -209.7961884, 811.2807007, -218.4419098, 843.8818970, -1053.6781006, 1029.7226562
3: -535.2235718, 780.4590454, -557.8200684, 812.7958374, -1348.0194092, 1338.2790527
4: -334.2570190, 833.0983276, -347.6372986, 866.6468506, -1200.9038086, 1180.7355957

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074905, upper bound: 843.2077624
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2082965, upper bound: 843.2073601
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2082965, upper bound: 843.2078159
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -185.4147034, 630.6639404, -191.5124207, 651.7287598, -837.1434326, 822.1763916
1: -302.0009155, 770.8518677, -313.5919189, 796.0115967, -1098.0122070, 1084.4436035
2: -211.2521362, 816.2529297, -217.9619141, 842.0349121, -1053.2871094, 1034.2148438
3: -538.4077759, 785.4230957, -556.5473633, 811.0206909, -1349.4283447, 1341.9704590
4: -336.6280823, 838.2955933, -346.8892517, 864.7709351, -1201.3989258, 1185.1848145

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2085123, upper bound: 843.2083257
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087240, upper bound: 843.2077503
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087240, upper bound: 843.2083269
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -188.3924713, 641.2564087, -829.7315674, 829.6968994
1: -308.6676941, 783.1837769, -307.3268738, 783.6063843, -1092.2739258, 1090.5106201
2: -214.4147797, 828.5700073, -214.6040497, 829.6708374, -1044.0855713, 1043.1740723
3: -547.7684326, 797.9298096, -547.3501587, 798.2861328, -1346.0541992, 1345.2800293
4: -341.2081909, 850.9271851, -341.8905029, 851.8900757, -1193.0981445, 1192.8176270

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2073601, upper bound: 843.2082965
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2073601, upper bound: 843.2087240
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -188.0576477, 640.1040649, -827.2272339, 824.2988281
1: -305.6914673, 777.5458984, -306.7427063, 782.2080688, -1087.8994141, 1084.2885742
2: -213.0952911, 822.5731201, -214.2190247, 828.1894531, -1041.2846680, 1036.7921143
3: -543.3413086, 792.1798706, -546.3464966, 796.8613892, -1340.2026367, 1338.5263672
4: -339.2044983, 844.9370728, -341.2821960, 850.3778076, -1189.5820312, 1186.2192383

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078159, upper bound: 843.2085432
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078159, upper bound: 843.2091336
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -191.9384003, 653.1663818, -841.6414795, 833.2428589
1: -308.6676941, 783.1837769, -314.3370667, 797.7437744, -1106.4111328, 1097.5208740
2: -214.4147797, 828.5700073, -218.4419098, 843.8818970, -1058.2966309, 1047.0119629
3: -547.7684326, 797.9298096, -557.8200684, 812.7958374, -1360.5639648, 1355.7497559
4: -341.2081909, 850.9271851, -347.6372986, 866.6468506, -1207.8549805, 1198.5644531

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063269, upper bound: 843.2063269
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063269, upper bound: 843.2067038
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -191.5124207, 651.7287598, -838.8518066, 827.7536621
1: -305.6914673, 777.5458984, -313.5919189, 796.0115967, -1101.7027588, 1091.1378174
2: -213.0952911, 822.5731201, -217.9619141, 842.0349121, -1055.1302490, 1040.5350342
3: -543.3413086, 792.1798706, -556.5473633, 811.0206909, -1354.3620605, 1348.7272949
4: -339.2044983, 844.9370728, -346.8892517, 864.7709351, -1203.9754639, 1191.8262939

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067038, upper bound: 843.2067933
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067038, upper bound: 843.2071776
time: 1.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.89 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2086150, upper bound: 843.2099345
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2086653, upper bound: 843.2091117
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2096367, upper bound: 843.2105097
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2096870, upper bound: 843.2096870
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2082965, upper bound: 843.2073601
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2082965, upper bound: 843.2078159
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2087240, upper bound: 843.2077503
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2087240, upper bound: 843.2083269
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2073601, upper bound: 843.2082965
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2073601, upper bound: 843.2087240
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2078159, upper bound: 843.2085432
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2078159, upper bound: 843.2091336
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2063269, upper bound: 843.2063269
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2063269, upper bound: 843.2067038
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2067038, upper bound: 843.2067933
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.89
Output dim: 0, lower bound: -843.2067038, upper bound: 843.2071776

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -184.1930847, 626.9782715, -171.6107483, 583.5519409, -767.7448730, 798.5889893
1: -300.4424438, 766.2285767, -280.5059509, 713.3177490, -1013.7601929, 1046.7344971
2: -209.7961884, 811.2807007, -195.5651093, 754.9420776, -964.7381592, 1006.8458252
3: -535.2235718, 780.4590454, -499.1630554, 726.9379272, -1262.1614990, 1279.6220703
4: -334.2570190, 833.0983276, -311.6126404, 775.2189941, -1109.4760742, 1144.7109375

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2086150, upper bound: 843.2091117
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2086150, upper bound: 843.2091117
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -182.9248047, 622.7667236, -208.0431061, 707.0399780, -889.9647827, 830.8098145
1: -298.3596802, 761.0418091, -342.9703064, 862.9215698, -1161.2812500, 1104.0120850
2: -208.3480530, 805.8724976, -237.0101166, 913.2547607, -1121.6026611, 1042.8825684
3: -531.5619507, 775.1286011, -607.4349976, 879.5144653, -1411.0762939, 1382.5634766
4: -331.9456787, 827.5151978, -376.8020630, 938.5618896, -1270.5072021, 1204.3168945

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2086653, upper bound: 843.2091117
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2086653, upper bound: 843.2091117
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -185.4147034, 630.6639404, -171.2514343, 582.3228760, -767.7375488, 801.9153442
1: -302.0009155, 770.8518677, -279.8710938, 711.8244629, -1013.8253784, 1050.7229004
2: -211.2521362, 816.2529297, -195.1512299, 753.3605347, -964.6126099, 1011.4041748
3: -538.4077759, 785.4230957, -498.0775146, 725.4132080, -1263.8208008, 1283.5006104
4: -336.6280823, 838.2955933, -310.9602356, 773.6007690, -1110.2283936, 1149.2558594

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096367, upper bound: 843.2096367
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096367, upper bound: 843.2096870
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -184.2378235, 626.7337036, -207.5772858, 705.4368286, -889.6746826, 834.3109741
1: -300.0782471, 766.0083008, -342.1478577, 860.9936523, -1161.0718994, 1108.1561279
2: -209.9072571, 811.2021484, -236.4815521, 911.2067871, -1121.1140137, 1047.6834717
3: -535.0038452, 780.4523315, -606.0353394, 877.5490723, -1412.5528564, 1386.4876709
4: -334.4808044, 833.0885010, -375.9789124, 936.4900513, -1270.9708252, 1209.0672607

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096870, upper bound: 843.2096367
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2096870, upper bound: 843.2096870
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -184.1930847, 626.9782715, -188.4751282, 641.3044434, -825.4973755, 815.4533691
1: -300.4424438, 766.2285767, -308.6676941, 783.1837769, -1083.6262207, 1074.8962402
2: -209.7961884, 811.2807007, -214.4147797, 828.5700073, -1038.3662109, 1025.6954346
3: -535.2235718, 780.4590454, -547.7684326, 797.9298096, -1333.1533203, 1328.2274170
4: -334.2570190, 833.0983276, -341.2081909, 850.9271851, -1185.1842041, 1174.3063965

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2082919, upper bound: 843.2072894
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072469, upper bound: 843.2069455
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -184.1930847, 626.9782715, -187.1232147, 636.2412109, -820.4341431, 814.1014404
1: -300.4424438, 766.2285767, -305.6914673, 777.5458984, -1077.9882812, 1071.9200439
2: -209.7961884, 811.2807007, -213.0952911, 822.5731201, -1032.3692627, 1024.3759766
3: -535.2235718, 780.4590454, -543.3413086, 792.1798706, -1327.4034424, 1323.8002930
4: -334.2570190, 833.0983276, -339.2044983, 844.9370728, -1179.1940918, 1172.3028564

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2082919, upper bound: 843.2077168
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072469, upper bound: 843.2074125
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -185.4147034, 630.6639404, -188.4751282, 641.3044434, -826.7191162, 819.1390381
1: -302.0009155, 770.8518677, -308.6676941, 783.1837769, -1085.1846924, 1079.5192871
2: -211.2521362, 816.2529297, -214.4147797, 828.5700073, -1039.8221436, 1030.6677246
3: -538.4077759, 785.4230957, -547.7684326, 797.9298096, -1336.3374023, 1333.1914062
4: -336.6280823, 838.2955933, -341.2081909, 850.9271851, -1187.5551758, 1179.5036621

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087237, upper bound: 843.2076697
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081832, upper bound: 843.2074822
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -185.4147034, 630.6639404, -187.1232147, 636.2412109, -821.6558838, 817.7871094
1: -302.0009155, 770.8518677, -305.6914673, 777.5458984, -1079.5468750, 1076.5432129
2: -211.2521362, 816.2529297, -213.0952911, 822.5731201, -1033.8251953, 1029.3481445
3: -538.4077759, 785.4230957, -543.3413086, 792.1798706, -1330.5875244, 1328.7644043
4: -336.6280823, 838.2955933, -339.2044983, 844.9370728, -1181.5648193, 1177.5001221

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087237, upper bound: 843.2076697
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081832, upper bound: 843.2080767
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -184.1930847, 626.9782715, -815.4533691, 825.4973755
1: -308.6676941, 783.1837769, -300.4424438, 766.2285767, -1074.8962402, 1083.6262207
2: -214.4147797, 828.5700073, -209.7961884, 811.2807007, -1025.6954346, 1038.3662109
3: -547.7684326, 797.9298096, -535.2235718, 780.4590454, -1328.2274170, 1333.1533203
4: -341.2081909, 850.9271851, -334.2570190, 833.0983276, -1174.3063965, 1185.1842041

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070689, upper bound: 843.2069849
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069531, upper bound: 843.2069929
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2072469
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -185.4147034, 630.6639404, -819.1390381, 826.7191162
1: -308.6676941, 783.1837769, -302.0009155, 770.8518677, -1079.5192871, 1085.1846924
2: -214.4147797, 828.5700073, -211.2521362, 816.2529297, -1030.6676025, 1039.8221436
3: -547.7684326, 797.9298096, -538.4077759, 785.4230957, -1333.1914062, 1336.3374023
4: -341.2081909, 850.9271851, -336.6280823, 838.2955933, -1179.5036621, 1187.5551758

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070689, upper bound: 843.2080072
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069531, upper bound: 843.2069929
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2081832
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -184.1930847, 626.9782715, -814.1014404, 820.4341431
1: -305.6914673, 777.5458984, -300.4424438, 766.2285767, -1071.9200439, 1077.9882812
2: -213.0952911, 822.5731201, -209.7961884, 811.2807007, -1024.3759766, 1032.3692627
3: -543.3413086, 792.1798706, -535.2235718, 780.4590454, -1323.8002930, 1327.4034424
4: -339.2044983, 844.9370728, -334.2570190, 833.0983276, -1172.3028564, 1179.1940918

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077624, upper bound: 843.2074905
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069531, upper bound: 843.2074592
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074125, upper bound: 843.2077203
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -185.4147034, 630.6639404, -817.7871094, 821.6558838
1: -305.6914673, 777.5458984, -302.0009155, 770.8518677, -1076.5432129, 1079.5468750
2: -213.0952911, 822.5731201, -211.2521362, 816.2529297, -1029.3481445, 1033.8251953
3: -543.3413086, 792.1798706, -538.4077759, 785.4230957, -1328.7644043, 1330.5875244
4: -339.2044983, 844.9370728, -336.6280823, 838.2955933, -1177.5001221, 1181.5648193

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077624, upper bound: 843.2082958
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072911, upper bound: 843.2082961
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074125, upper bound: 843.2085774
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -188.4751282, 641.3044434, -829.7795410, 829.7795410
1: -308.6676941, 783.1837769, -308.6676941, 783.1837769, -1091.8514404, 1091.8514404
2: -214.4147797, 828.5700073, -214.4147797, 828.5700073, -1042.9847412, 1042.9847412
3: -547.7684326, 797.9298096, -547.7684326, 797.9298096, -1345.6978760, 1345.6978760
4: -341.2081909, 850.9271851, -341.2081909, 850.9271851, -1192.1352539, 1192.1352539

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2062458, upper bound: 843.2060439
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2062406, upper bound: 843.2062406
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -187.1232147, 636.2412109, -824.7163086, 828.4276123
1: -308.6676941, 783.1837769, -305.6914673, 777.5458984, -1086.2135010, 1088.8752441
2: -214.4147797, 828.5700073, -213.0952911, 822.5731201, -1036.9879150, 1041.6652832
3: -547.7684326, 797.9298096, -543.3413086, 792.1798706, -1339.9479980, 1341.2711182
4: -341.2081909, 850.9271851, -339.2044983, 844.9370728, -1186.1448975, 1190.1317139

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2062458, upper bound: 843.2063650
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2062406, upper bound: 843.2066186
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -188.4751282, 641.3044434, -828.4276123, 824.7163086
1: -305.6914673, 777.5458984, -308.6676941, 783.1837769, -1088.8752441, 1086.2135010
2: -213.0952911, 822.5731201, -214.4147797, 828.5700073, -1041.6652832, 1036.9879150
3: -543.3413086, 792.1798706, -547.7684326, 797.9298096, -1341.2711182, 1339.9479980
4: -339.2044983, 844.9370728, -341.2081909, 850.9271851, -1190.1317139, 1186.1448975

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065372, upper bound: 843.2065075
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2066186, upper bound: 843.2067144
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -187.1232147, 636.2412109, -823.3643799, 823.3643799
1: -305.6914673, 777.5458984, -305.6914673, 777.5458984, -1083.2373047, 1083.2373047
2: -213.0952911, 822.5731201, -213.0952911, 822.5731201, -1035.6683350, 1035.6684570
3: -543.3413086, 792.1798706, -543.3413086, 792.1798706, -1335.5212402, 1335.5212402
4: -339.2044983, 844.9370728, -339.2044983, 844.9370728, -1184.1414795, 1184.1414795

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065372, upper bound: 843.2065076
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2066186, upper bound: 843.2070937
time: 1.08 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.52 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2086150, upper bound: 843.2091117
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2086150, upper bound: 843.2091117
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2086653, upper bound: 843.2091117
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2086653, upper bound: 843.2091117
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2096367, upper bound: 843.2096367
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2096367, upper bound: 843.2096870
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2096870, upper bound: 843.2096367
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2096870, upper bound: 843.2096870
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2082919, upper bound: 843.2072894
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2072469, upper bound: 843.2069455
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2082919, upper bound: 843.2077168
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2072469, upper bound: 843.2074125
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2087237, upper bound: 843.2076697
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2081832, upper bound: 843.2074822
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2087237, upper bound: 843.2076697
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2081832, upper bound: 843.2080767
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2069531, upper bound: 843.2069929
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2072469
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2069531, upper bound: 843.2069929
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2081832
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2069531, upper bound: 843.2074592
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2074125, upper bound: 843.2077203
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2072911, upper bound: 843.2082961
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2074125, upper bound: 843.2085774
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2062458, upper bound: 843.2060439
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2062406, upper bound: 843.2062406
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2062458, upper bound: 843.2063650
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2062406, upper bound: 843.2066186
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2065372, upper bound: 843.2065075
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2066186, upper bound: 843.2067144
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2065372, upper bound: 843.2065076
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -843.2066186, upper bound: 843.2070937

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -167.5513153, 569.7696533, -171.6107483, 583.5519409, -751.1032104, 741.3803711
1: -273.8763428, 696.5308838, -280.5059509, 713.3177490, -987.1939697, 977.0368652
2: -190.9233856, 737.1707153, -195.5651093, 754.9420776, -945.8654785, 932.7358398
3: -487.4682312, 709.7176514, -499.1630554, 726.9379272, -1214.4061279, 1208.8807373
4: -304.2389221, 757.0684814, -311.6126404, 775.2189941, -1079.4577637, 1068.6811523

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2096245
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2099345
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -204.7340698, 695.8717041, -171.6107483, 583.5519409, -788.2860107, 867.4824219
1: -337.6034241, 849.2514038, -280.5059509, 713.3177490, -1050.9211426, 1129.7573242
2: -233.1838074, 898.7626343, -195.5651093, 754.9420776, -988.1258545, 1094.3277588
3: -597.8302002, 865.3538818, -499.1630554, 726.9379272, -1324.7680664, 1364.5169678
4: -370.6482544, 923.5570679, -311.6126404, 775.2189941, -1145.8671875, 1235.1696777

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2096245
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2099345
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -167.5513153, 569.7696533, -208.0431061, 707.0399780, -874.5912476, 777.8126831
1: -273.8763428, 696.5308838, -342.9703064, 862.9215698, -1136.7978516, 1039.5009766
2: -190.9233856, 737.1707153, -237.0101166, 913.2547607, -1104.1779785, 974.1808472
3: -487.4682312, 709.7176514, -607.4349976, 879.5144653, -1366.9826660, 1317.1525879
4: -304.2389221, 757.0684814, -376.8020630, 938.5618896, -1242.8005371, 1133.8704834

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -204.7340698, 695.8717041, -208.0431061, 707.0399780, -911.7740479, 903.9147949
1: -337.6034241, 849.2514038, -342.9703064, 862.9215698, -1200.5250244, 1192.2214355
2: -233.1838074, 898.7626343, -237.0101166, 913.2547607, -1146.4384766, 1135.7727051
3: -597.8302002, 865.3538818, -607.4349976, 879.5144653, -1477.3446045, 1472.7888184
4: -370.6482544, 923.5570679, -376.8020630, 938.5618896, -1309.2099609, 1300.3590088

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -168.3289490, 571.9808350, -171.2514343, 582.3228760, -750.6517944, 743.2322388
1: -274.5803833, 699.3826294, -279.8710938, 711.8244629, -986.4048462, 979.2537231
2: -191.8508453, 740.2653198, -195.1512299, 753.3605347, -945.2113037, 935.4164429
3: -489.2566528, 712.8433838, -498.0775146, 725.4132080, -1214.6696777, 1210.9205322
4: -305.7969666, 760.3158569, -310.9602356, 773.6007690, -1079.3975830, 1071.2760010

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091492, upper bound: 843.2101494
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091492, upper bound: 843.2104595
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -201.7331848, 684.9160767, -171.2514343, 582.3228760, -784.0560303, 856.1674805
1: -331.7389526, 836.4715576, -279.8710938, 711.8244629, -1043.5633545, 1116.3426514
2: -229.9456787, 885.3747559, -195.1512299, 753.3605347, -983.3062134, 1080.5260010
3: -588.3772583, 852.6589355, -498.0775146, 725.4132080, -1313.7905273, 1350.7360840
4: -365.7247314, 910.3051758, -310.9602356, 773.6007690, -1139.3254395, 1221.2653809

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091492, upper bound: 843.2101997
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091492, upper bound: 843.2101997
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -168.3289490, 571.9808350, -207.5772858, 705.4368286, -873.7657471, 779.5579834
1: -274.5803833, 699.3826294, -342.1478577, 860.9936523, -1135.5739746, 1041.5305176
2: -191.8508453, 740.2653198, -236.4815521, 911.2067871, -1103.0576172, 976.7468262
3: -489.2566528, 712.8433838, -606.0353394, 877.5490723, -1366.8052979, 1318.8786621
4: -305.7969666, 760.3158569, -375.9789124, 936.4900513, -1242.2869873, 1136.2945557

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2086150
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2096163
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -201.7331848, 684.9160767, -207.5772858, 705.4368286, -907.1700439, 892.4932861
1: -331.7389526, 836.4715576, -342.1478577, 860.9936523, -1192.7326660, 1178.6193848
2: -229.9456787, 885.3747559, -236.4815521, 911.2067871, -1141.1524658, 1121.8563232
3: -588.3772583, 852.6589355, -606.0353394, 877.5490723, -1465.9262695, 1458.6942139
4: -365.7247314, 910.3051758, -375.9789124, 936.4900513, -1302.2148438, 1286.2840576

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2086653
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2096852
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -172.2730255, 587.1818848, -187.5928802, 638.4122925, -810.6853027, 774.7746582
1: -280.8097839, 717.5310059, -307.1780090, 779.6473389, -1060.4567871, 1024.7089844
2: -196.2395630, 759.6019897, -213.4097443, 824.8283691, -1021.0679321, 973.0116577
3: -500.6431580, 730.5495605, -545.1756592, 794.2838745, -1294.9270020, 1275.7252197
4: -312.7213440, 779.9223633, -339.6175842, 847.0654907, -1159.7868652, 1119.5397949

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069783, upper bound: 843.2069467
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069929, upper bound: 843.2069455
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069929, upper bound: 843.2069455
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -179.2961731, 610.8264160, -188.2920532, 640.6959839, -819.9920654, 799.1184692
1: -292.4148254, 746.4031372, -308.3717651, 782.4371948, -1074.8520508, 1054.7747803
2: -204.1958008, 790.3706055, -214.2057800, 827.7813721, -1031.9771729, 1004.5762939
3: -521.0043945, 760.0877075, -547.2412720, 797.1643677, -1318.1687012, 1307.3286133
4: -325.3488159, 811.5093994, -340.8743286, 850.1145020, -1175.4633789, 1152.3835449

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2064239, upper bound: 843.2066726
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069929, upper bound: 843.2069455
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069929, upper bound: 843.2069455
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -172.2730255, 587.1818848, -186.2418213, 633.3479614, -805.6209106, 773.4237061
1: -280.8097839, 717.5310059, -304.2053223, 774.0137329, -1054.8233643, 1021.7363281
2: -196.2395630, 759.6019897, -212.0907288, 818.8385620, -1015.0781250, 971.6927490
3: -500.6431580, 730.5495605, -540.7645874, 788.5366211, -1289.1795654, 1271.3138428
4: -312.7213440, 779.9223633, -337.6135864, 841.0750122, -1153.7963867, 1117.5358887

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074895, upper bound: 843.2076334
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074592, upper bound: 843.2072911
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074592, upper bound: 843.2074125
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -179.2961731, 610.8264160, -186.9377594, 635.6281738, -814.9243164, 797.7640991
1: -292.4148254, 746.4031372, -305.3922119, 776.7917480, -1069.2065430, 1051.7952881
2: -204.1958008, 790.3706055, -212.8829803, 821.7766724, -1025.9724121, 1003.2536011
3: -521.0043945, 760.0877075, -542.8035889, 791.4058838, -1312.4102783, 1302.8909912
4: -325.3488159, 811.5093994, -338.8660889, 844.1160889, -1169.4648438, 1150.3753662

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069372, upper bound: 843.2074125
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074592, upper bound: 843.2072911
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074592, upper bound: 843.2074125
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -173.9681702, 592.3180542, -187.5928802, 638.4122925, -812.3804932, 779.9108887
1: -283.2098389, 723.9351807, -307.1780090, 779.6473389, -1062.8569336, 1031.1131592
2: -198.2450714, 766.4570923, -213.4097443, 824.8283691, -1023.0734253, 979.8667603
3: -505.2448425, 737.4182129, -545.1756592, 794.2838745, -1299.5286865, 1282.5938721
4: -315.9694824, 787.1150513, -339.6175842, 847.0654907, -1163.0349121, 1126.7325439

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079994, upper bound: 843.2074845
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078875, upper bound: 843.2074459
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078875, upper bound: 843.2074822
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -180.2173462, 613.6436768, -188.2920532, 640.6959839, -820.9133301, 801.9357300
1: -293.4556274, 749.9676514, -308.3717651, 782.4371948, -1075.8928223, 1058.3393555
2: -205.3023071, 794.1782837, -214.2057800, 827.7813721, -1033.0837402, 1008.3840332
3: -523.2786865, 763.9072266, -547.2412720, 797.1643677, -1320.4431152, 1311.1479492
4: -327.1666565, 815.4519653, -340.8743286, 850.1145020, -1177.2811279, 1156.3259277

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079290, upper bound: 843.2073583
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078875, upper bound: 843.2074459
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078875, upper bound: 843.2074822
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -173.9681702, 592.3180542, -186.2418213, 633.3479614, -807.3161011, 778.5598755
1: -283.2098389, 723.9351807, -304.2053223, 774.0137329, -1057.2233887, 1028.1405029
2: -198.2450714, 766.4570923, -212.0907288, 818.8385620, -1017.0836182, 978.5478516
3: -505.2448425, 737.4182129, -540.7645874, 788.5366211, -1293.7814941, 1278.1824951
4: -315.9694824, 787.1150513, -337.6135864, 841.0750122, -1157.0444336, 1124.7285156

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2085115, upper bound: 843.2082034
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2083972, upper bound: 843.2079269
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069929, upper bound: 843.2080767
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -180.2173462, 613.6436768, -186.9377594, 635.6281738, -815.8455200, 800.5814209
1: -293.4556274, 749.9676514, -305.3922119, 776.7917480, -1070.2473145, 1055.3598633
2: -205.3023071, 794.1782837, -212.8829803, 821.7766724, -1027.0789795, 1007.0612793
3: -523.2786865, 763.9072266, -542.8035889, 791.4058838, -1314.6845703, 1306.7105713
4: -327.1666565, 815.4519653, -338.8660889, 844.1160889, -1171.2827148, 1154.3177490

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2084432, upper bound: 843.2080762
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2083972, upper bound: 843.2079269
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2083972, upper bound: 843.2080767
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -175.5612030, 598.3837891, -183.3984833, 624.3621216, -799.9233398, 781.7821045
1: -287.3129883, 730.6085205, -299.1100159, 763.0281372, -1050.3410645, 1029.7185059
2: -199.7297058, 772.8367310, -208.8870087, 807.8942261, -1007.6239014, 981.7236938
3: -510.0922241, 743.9570923, -532.8960571, 777.1630859, -1287.2552490, 1276.8531494
4: -317.8653870, 793.4354248, -332.8158875, 829.5971680, -1147.4625244, 1126.2513428

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2069929
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2069929
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -184.1430817, 626.8763428, -183.9839630, 626.2910767, -810.4341431, 810.8602905
1: -301.7005615, 765.4571533, -300.0999146, 765.3843384, -1067.0848389, 1065.5571289
2: -209.4716339, 809.8760986, -209.5571594, 810.3903809, -1019.8619995, 1019.4332275
3: -535.3128052, 779.7720947, -534.6160889, 779.5904541, -1314.9031982, 1314.3881836
4: -333.3037109, 831.6660767, -333.8762817, 832.1776733, -1165.4813232, 1165.5423584

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2072469
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2072469
time: 3.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -175.5612030, 598.3837891, -184.6471405, 628.1339111, -803.6951294, 783.0309448
1: -287.3129883, 730.6085205, -300.7195740, 767.7531738, -1055.0661621, 1031.3281250
2: -199.7297058, 772.8367310, -210.3744812, 812.9765015, -1012.7061768, 983.2111206
3: -510.0922241, 743.9570923, -536.1627197, 782.2310181, -1292.3232422, 1280.1198730
4: -317.8653870, 793.4354248, -335.2355042, 834.9113159, -1152.7766113, 1128.6706543

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074459, upper bound: 843.2078875
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074459, upper bound: 843.2078875
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -184.1430817, 626.8763428, -185.2000732, 629.9617920, -814.1048584, 812.0764160
1: -301.7005615, 765.4571533, -301.6477966, 769.9901733, -1071.6906738, 1067.1049805
2: -209.4716339, 809.8760986, -211.0066223, 815.3428345, -1024.8144531, 1020.8826294
3: -535.3128052, 779.7720947, -537.7832642, 784.5354614, -1319.8482666, 1317.5554199
4: -333.3037109, 831.6660767, -336.2376099, 837.3535156, -1170.6572266, 1167.9036865

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074822, upper bound: 843.2081832
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074822, upper bound: 843.2081832
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -174.3998871, 593.9743042, -183.3984833, 624.3621216, -798.7619629, 777.3726196
1: -284.6991577, 725.7923584, -299.1100159, 763.0281372, -1047.7271729, 1024.9023438
2: -198.6237030, 767.6218262, -208.8870087, 807.8942261, -1006.5179443, 976.5088501
3: -506.3637085, 739.0524292, -532.8960571, 777.1630859, -1283.5264893, 1271.9484863
4: -316.1923523, 788.3237915, -332.8158875, 829.5971680, -1145.7893066, 1121.1396484

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072911, upper bound: 843.2074592
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072911, upper bound: 843.2074592
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -182.8314972, 621.9671631, -183.9839630, 626.2910767, -809.1225586, 805.9511108
1: -298.8067932, 759.9968262, -300.0999146, 765.3843384, -1064.1910400, 1060.0966797
2: -208.1809540, 804.0384521, -209.5571594, 810.3903809, -1018.5713501, 1013.5955811
3: -530.9179077, 774.1968384, -534.6160889, 779.5904541, -1310.5081787, 1308.8128662
4: -331.3768921, 825.8750000, -333.8762817, 832.1776733, -1163.5545654, 1159.7512207

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074125, upper bound: 843.2077203
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2074125, upper bound: 843.2077203
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -174.3998871, 593.9743042, -184.6471405, 628.1339111, -802.5337524, 778.6214600
1: -284.6991577, 725.7923584, -300.7195740, 767.7531738, -1052.4521484, 1026.5119629
2: -198.6237030, 767.6218262, -210.3744812, 812.9765015, -1011.6002197, 977.9963379
3: -506.3637085, 739.0524292, -536.1627197, 782.2310181, -1288.5947266, 1275.2149658
4: -316.1923523, 788.3237915, -335.2355042, 834.9113159, -1151.1033936, 1123.5589600

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2075455, upper bound: 843.2082961
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2075455, upper bound: 843.2082961
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -182.8314972, 621.9671631, -185.2000732, 629.9617920, -812.7932739, 807.1671753
1: -298.8067932, 759.9968262, -301.6477966, 769.9901733, -1068.7969971, 1061.6446533
2: -208.1809540, 804.0384521, -211.0066223, 815.3428345, -1023.5238037, 1015.0449829
3: -530.9179077, 774.1968384, -537.7832642, 784.5354614, -1315.4533691, 1311.9801025
4: -331.3768921, 825.8750000, -336.2376099, 837.3535156, -1168.7303467, 1162.1125488

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078378, upper bound: 843.2085774
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2078378, upper bound: 843.2085774
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -175.5612030, 598.3837891, -187.5928802, 638.4122925, -813.9735107, 785.9766235
1: -287.3129883, 730.6085205, -307.1780090, 779.6473389, -1066.9602051, 1037.7864990
2: -199.7297058, 772.8367310, -213.4097443, 824.8283691, -1024.5579834, 986.2463989
3: -510.0922241, 743.9570923, -545.1756592, 794.2838745, -1304.3760986, 1289.1328125
4: -317.8653870, 793.4354248, -339.6175842, 847.0654907, -1164.9309082, 1133.0529785

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060440, upper bound: 843.2060440
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060440, upper bound: 843.2060440
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -184.1430817, 626.8763428, -188.2920532, 640.6959839, -824.8390503, 815.1683960
1: -301.7005615, 765.4571533, -308.3717651, 782.4371948, -1084.1376953, 1073.8288574
2: -209.4716339, 809.8760986, -214.2057800, 827.7813721, -1037.2530518, 1024.0817871
3: -535.3128052, 779.7720947, -547.2412720, 797.1643677, -1332.4771729, 1327.0130615
4: -333.3037109, 831.6660767, -340.8743286, 850.1145020, -1183.4182129, 1172.5402832

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060440, upper bound: 843.2062406
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060440, upper bound: 843.2062406
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -175.5612030, 598.3837891, -186.2418213, 633.3479614, -808.9091187, 784.6256104
1: -287.3129883, 730.6085205, -304.2053223, 774.0137329, -1061.3266602, 1034.8137207
2: -199.7297058, 772.8367310, -212.0907288, 818.8385620, -1018.5682373, 984.9274292
3: -510.0922241, 743.9570923, -540.7645874, 788.5366211, -1298.6289062, 1284.7216797
4: -317.8653870, 793.4354248, -337.6135864, 841.0750122, -1158.9404297, 1131.0489502

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065075, upper bound: 843.2062903
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065075, upper bound: 843.2063650
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -184.1430817, 626.8763428, -186.9377594, 635.6281738, -819.7712402, 813.8140869
1: -301.7005615, 765.4571533, -305.3922119, 776.7917480, -1078.4923096, 1070.8493652
2: -209.4716339, 809.8760986, -212.8829803, 821.7766724, -1031.2482910, 1022.7590942
3: -535.3128052, 779.7720947, -542.8035889, 791.4058838, -1326.7187500, 1322.5755615
4: -333.3037109, 831.6660767, -338.8660889, 844.1160889, -1177.4197998, 1170.5321045

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065075, upper bound: 843.2065372
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065075, upper bound: 843.2066186
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -174.3998871, 593.9743042, -187.5928802, 638.4122925, -812.8121338, 781.5671387
1: -284.6991577, 725.7923584, -307.1780090, 779.6473389, -1064.3460693, 1032.9703369
2: -198.6237030, 767.6218262, -213.4097443, 824.8283691, -1023.4520874, 981.0315552
3: -506.3637085, 739.0524292, -545.1756592, 794.2838745, -1300.6475830, 1284.2280273
4: -316.1923523, 788.3237915, -339.6175842, 847.0654907, -1163.2578125, 1127.9412842

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2062903, upper bound: 843.2065075
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2062903, upper bound: 843.2065075
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -182.8314972, 621.9671631, -188.2920532, 640.6959839, -823.5274048, 810.2591553
1: -298.8067932, 759.9968262, -308.3717651, 782.4371948, -1081.2440186, 1068.3686523
2: -208.1809540, 804.0384521, -214.2057800, 827.7813721, -1035.9622803, 1018.2441406
3: -530.9179077, 774.1968384, -547.2412720, 797.1643677, -1328.0822754, 1321.4378662
4: -331.3768921, 825.8750000, -340.8743286, 850.1145020, -1181.4914551, 1166.7492676

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063650, upper bound: 843.2066656
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063650, upper bound: 843.2067144
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -174.3998871, 593.9743042, -186.2418213, 633.3479614, -807.7477417, 780.2161255
1: -284.6991577, 725.7923584, -304.2053223, 774.0137329, -1058.7125244, 1029.9974365
2: -198.6237030, 767.6218262, -212.0907288, 818.8385620, -1017.4622803, 979.7125244
3: -506.3637085, 739.0524292, -540.7645874, 788.5366211, -1294.9000244, 1279.8167725
4: -316.1923523, 788.3237915, -337.6135864, 841.0750122, -1157.2673340, 1125.9372559

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065902, upper bound: 843.2067773
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065902, upper bound: 843.2068499
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -182.8314972, 621.9671631, -186.9377594, 635.6281738, -818.4596558, 808.9048462
1: -298.8067932, 759.9968262, -305.3922119, 776.7917480, -1075.5985107, 1065.3890381
2: -208.1809540, 804.0384521, -212.8829803, 821.7766724, -1029.9576416, 1016.9214478
3: -530.9179077, 774.1968384, -542.8035889, 791.4058838, -1322.3237305, 1317.0003662
4: -331.3768921, 825.8750000, -338.8660889, 844.1160889, -1175.4929199, 1164.7409668

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067781, upper bound: 843.2070111
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067781, upper bound: 843.2070937
time: 0.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.31 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2096245
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2099345
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2096245
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2081305, upper bound: 843.2099345
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2080900, upper bound: 843.2080900
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2091492, upper bound: 843.2101494
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2091492, upper bound: 843.2104595
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2091492, upper bound: 843.2101997
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2091492, upper bound: 843.2101997
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2086150
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2096163
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2086653
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2091117, upper bound: 843.2096852
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2069929, upper bound: 843.2069455
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2069929, upper bound: 843.2069455
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2069929, upper bound: 843.2069455
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2069929, upper bound: 843.2069455
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2074592, upper bound: 843.2072911
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2074592, upper bound: 843.2074125
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2074592, upper bound: 843.2072911
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2074592, upper bound: 843.2074125
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2078875, upper bound: 843.2074459
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2078875, upper bound: 843.2074822
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2078875, upper bound: 843.2074459
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2078875, upper bound: 843.2074822
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2083972, upper bound: 843.2079269
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2069929, upper bound: 843.2080767
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2083972, upper bound: 843.2079269
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2083972, upper bound: 843.2080767
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2069929
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2069929
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2072469
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2072469
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2074459, upper bound: 843.2078875
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2074459, upper bound: 843.2078875
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2074822, upper bound: 843.2081832
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2074822, upper bound: 843.2081832
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2072911, upper bound: 843.2074592
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2072911, upper bound: 843.2074592
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2074125, upper bound: 843.2077203
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2074125, upper bound: 843.2077203
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2075455, upper bound: 843.2082961
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2075455, upper bound: 843.2082961
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2078378, upper bound: 843.2085774
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2078378, upper bound: 843.2085774
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2060440, upper bound: 843.2060440
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2060440, upper bound: 843.2060440
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2060440, upper bound: 843.2062406
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2060440, upper bound: 843.2062406
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2065075, upper bound: 843.2062903
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2065075, upper bound: 843.2063650
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2065075, upper bound: 843.2065372
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2065075, upper bound: 843.2066186
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2062903, upper bound: 843.2065075
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2062903, upper bound: 843.2065075
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2063650, upper bound: 843.2066656
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2063650, upper bound: 843.2067144
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2065902, upper bound: 843.2067773
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2065902, upper bound: 843.2068499
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2067781, upper bound: 843.2070111
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.31
Output dim: 0, lower bound: -843.2067781, upper bound: 843.2070937

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -167.5513153, 569.7696533, -167.5513153, 569.7696533, -737.3209229, 737.3209229
1: -273.8763428, 696.5308838, -273.8763428, 696.5308838, -970.4071655, 970.4071655
2: -190.9233856, 737.1707153, -190.9233856, 737.1707153, -928.0941162, 928.0941162
3: -487.4682312, 709.7176514, -487.4682312, 709.7176514, -1197.1859131, 1197.1859131
4: -304.2389221, 757.0684814, -304.2389221, 757.0684814, -1061.3073730, 1061.3073730

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091608, upper bound: 843.2083712
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079872, upper bound: 843.2079872
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -167.5513153, 569.7696533, -168.3289490, 571.9808350, -739.5321045, 738.0985718
1: -273.8763428, 696.5308838, -274.5803833, 699.3826294, -973.2587891, 971.1112671
2: -190.9233856, 737.1707153, -191.8508453, 740.2653198, -931.1887207, 929.0214844
3: -487.4682312, 709.7176514, -489.2566528, 712.8433838, -1200.3115234, 1198.9742432
4: -304.2389221, 757.0684814, -305.7969666, 760.3158569, -1064.5545654, 1062.8654785

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091608, upper bound: 843.2092864
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079872, upper bound: 843.2089284
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -204.7340698, 695.8717041, -167.5513153, 569.7696533, -774.5037231, 863.4230347
1: -337.6034241, 849.2514038, -273.8763428, 696.5308838, -1034.1342773, 1123.1276855
2: -233.1838074, 898.7626343, -190.9233856, 737.1707153, -970.3544922, 1089.6859131
3: -597.8302002, 865.3538818, -487.4682312, 709.7176514, -1307.5478516, 1352.8221436
4: -370.6482544, 923.5570679, -304.2389221, 757.0684814, -1127.7167969, 1227.7958984

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2083244
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2080793
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -204.7340698, 695.8717041, -168.3289490, 571.9808350, -776.7149048, 864.2006836
1: -337.6034241, 849.2514038, -274.5803833, 699.3826294, -1036.9860840, 1123.8317871
2: -233.1838074, 898.7626343, -191.8508453, 740.2653198, -973.4490356, 1090.6135254
3: -597.8302002, 865.3538818, -489.2566528, 712.8433838, -1310.6733398, 1354.6104736
4: -370.6482544, 923.5570679, -305.7969666, 760.3158569, -1130.9639893, 1229.3540039

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2092646
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2090248
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -167.5513153, 569.7696533, -204.7340698, 695.8717041, -863.4230347, 774.5037231
1: -273.8763428, 696.5308838, -337.6034241, 849.2514038, -1123.1276855, 1034.1342773
2: -190.9233856, 737.1707153, -233.1838074, 898.7626343, -1089.6860352, 970.3544922
3: -487.4682312, 709.7176514, -597.8302002, 865.3538818, -1352.8221436, 1307.5478516
4: -304.2389221, 757.0684814, -370.6482544, 923.5570679, -1227.7958984, 1127.7167969

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091451, upper bound: 843.2075373
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080793, upper bound: 843.2072034
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -167.5513153, 569.7696533, -202.6466675, 687.9989624, -855.5502319, 772.4163208
1: -273.8763428, 696.5308838, -333.1848145, 840.2247314, -1114.1009521, 1029.7156982
2: -190.9233856, 737.1707153, -230.9812775, 889.4022217, -1080.3255615, 968.1519775
3: -487.4682312, 709.7176514, -591.0267944, 856.4942627, -1343.9624023, 1300.7443848
4: -304.2389221, 757.0684814, -367.3989258, 914.4356689, -1218.6745605, 1124.4674072

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091451, upper bound: 843.2090131
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080793, upper bound: 843.2087032
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -204.7340698, 695.8717041, -204.7340698, 695.8717041, -900.6057739, 900.6057739
1: -337.6034241, 849.2514038, -337.6034241, 849.2514038, -1186.8547363, 1186.8547363
2: -233.1838074, 898.7626343, -233.1838074, 898.7626343, -1131.9464111, 1131.9464111
3: -597.8302002, 865.3538818, -597.8302002, 865.3538818, -1463.1840820, 1463.1840820
4: -370.6482544, 923.5570679, -370.6482544, 923.5570679, -1294.2053223, 1294.2053223

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2075300
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2073002
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -204.7340698, 695.8717041, -202.6466675, 687.9989624, -892.7330322, 898.5183716
1: -337.6034241, 849.2514038, -333.1848145, 840.2247314, -1177.8278809, 1182.4360352
2: -233.1838074, 898.7626343, -230.9812775, 889.4022217, -1122.5860596, 1129.7437744
3: -597.8302002, 865.3538818, -591.0267944, 856.4942627, -1454.3242188, 1456.3806152
4: -370.6482544, 923.5570679, -367.3989258, 914.4356689, -1285.0839844, 1290.9560547

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2090258
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2088000
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -168.3289490, 571.9808350, -167.5513153, 569.7696533, -738.0985718, 739.5321045
1: -274.5803833, 699.3826294, -273.8763428, 696.5308838, -971.1112671, 973.2587891
2: -191.8508453, 740.2653198, -190.9233856, 737.1707153, -929.0214844, 931.1887207
3: -489.2566528, 712.8433838, -487.4682312, 709.7176514, -1198.9742432, 1200.3115234
4: -305.7969666, 760.3158569, -304.2389221, 757.0684814, -1062.8654785, 1064.5546875

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2095234, upper bound: 843.2088535
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2089284, upper bound: 843.2086440
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -168.3289490, 571.9808350, -168.3289490, 571.9808350, -740.3097534, 740.3097534
1: -274.5803833, 699.3826294, -274.5803833, 699.3826294, -973.9630127, 973.9630127
2: -191.8508453, 740.2653198, -191.8508453, 740.2653198, -932.1160889, 932.1160278
3: -489.2566528, 712.8433838, -489.2566528, 712.8433838, -1202.0997314, 1202.0997314
4: -305.7969666, 760.3158569, -305.7969666, 760.3158569, -1066.1127930, 1066.1127930

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2095234, upper bound: 843.2097934
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2089284, upper bound: 843.2095897
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -201.7331848, 684.9160767, -167.5513153, 569.7696533, -771.5028076, 852.4674072
1: -331.7389526, 836.4715576, -273.8763428, 696.5308838, -1028.2697754, 1110.3479004
2: -229.9456787, 885.3747559, -190.9233856, 737.1707153, -967.1163940, 1076.2980957
3: -588.3772583, 852.6589355, -487.4682312, 709.7176514, -1298.0949707, 1340.1270752
4: -365.7247314, 910.3051758, -304.2389221, 757.0684814, -1122.7932129, 1214.5440674

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2089064
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2087497
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -201.7331848, 684.9160767, -168.3289490, 571.9808350, -773.7139893, 853.2449951
1: -331.7389526, 836.4715576, -274.5803833, 699.3826294, -1031.1215820, 1111.0520020
2: -229.9456787, 885.3747559, -191.8508453, 740.2653198, -970.2109985, 1077.2255859
3: -588.3772583, 852.6589355, -489.2566528, 712.8433838, -1301.2205811, 1341.9152832
4: -365.7247314, 910.3051758, -305.7969666, 760.3158569, -1126.0405273, 1216.1020508

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2098466
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2096914
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -168.3289490, 571.9808350, -204.7340698, 695.8717041, -864.2006836, 776.7149048
1: -274.5803833, 699.3826294, -337.6034241, 849.2514038, -1123.8317871, 1036.9859619
2: -191.8508453, 740.2653198, -233.1838074, 898.7626343, -1090.6135254, 973.4490967
3: -489.2566528, 712.8433838, -597.8302002, 865.3538818, -1354.6104736, 1310.6733398
4: -305.7969666, 760.3158569, -370.6482544, 923.5570679, -1229.3540039, 1130.9639893

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2095383, upper bound: 843.2080587
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2090248, upper bound: 843.2078651
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -168.3289490, 571.9808350, -202.6466675, 687.9989624, -856.3278809, 774.6275024
1: -274.5803833, 699.3826294, -333.1848145, 840.2247314, -1114.8050537, 1032.5673828
2: -191.8508453, 740.2653198, -230.9812775, 889.4022217, -1081.2530518, 971.2465820
3: -489.2566528, 712.8433838, -591.0267944, 856.4942627, -1345.7504883, 1303.8698730
4: -305.7969666, 760.3158569, -367.3989258, 914.4356689, -1220.2326660, 1127.7147217

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2095384, upper bound: 843.2094802
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2090248, upper bound: 843.2093649
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -201.7331848, 684.9160767, -204.7340698, 695.8717041, -897.6048584, 889.6501465
1: -331.7389526, 836.4715576, -337.6034241, 849.2514038, -1180.9902344, 1174.0749512
2: -229.9456787, 885.3747559, -233.1838074, 898.7626343, -1128.7082520, 1118.5585938
3: -588.3772583, 852.6589355, -597.8302002, 865.3538818, -1453.7312012, 1450.4890137
4: -365.7247314, 910.3051758, -370.6482544, 923.5570679, -1289.2817383, 1280.9533691

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2081215
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2079665
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -201.7331848, 684.9160767, -202.6466675, 687.9989624, -889.7321777, 887.5627441
1: -331.7389526, 836.4715576, -333.1848145, 840.2247314, -1171.9636230, 1169.6562500
2: -229.9456787, 885.3747559, -230.9812775, 889.4022217, -1119.3479004, 1116.3559570
3: -588.3772583, 852.6589355, -591.0267944, 856.4942627, -1444.8715820, 1443.6854248
4: -365.7247314, 910.3051758, -367.3989258, 914.4356689, -1280.1604004, 1277.7041016

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2081215
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2094663
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -172.2730255, 587.1818848, -175.5612030, 598.3837891, -770.6567993, 762.7430420
1: -280.8097839, 717.5310059, -287.3129883, 730.6085205, -1011.4182739, 1004.8439331
2: -196.2395630, 759.6019897, -199.7297058, 772.8367310, -969.0762939, 959.3316650
3: -500.6431580, 730.5495605, -510.0922241, 743.9570923, -1244.6002197, 1240.6418457
4: -312.7213440, 779.9223633, -317.8653870, 793.4354248, -1106.1567383, 1097.7877197

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063006, upper bound: 843.2061507
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079693, upper bound: 843.2065698
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -172.2730255, 587.1818848, -184.1430817, 626.8763428, -799.1493530, 771.3249512
1: -280.8097839, 717.5310059, -301.7005615, 765.4571533, -1046.2668457, 1019.2315063
2: -196.2395630, 759.6019897, -209.4716339, 809.8760986, -1006.1156616, 969.0736084
3: -500.6431580, 730.5495605, -535.3128052, 779.7720947, -1280.4151611, 1265.8623047
4: -312.7213440, 779.9223633, -333.3037109, 831.6660767, -1144.3874512, 1113.2259521

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063006, upper bound: 843.2061507
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079693, upper bound: 843.2065698
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -179.2961731, 610.8264160, -175.5612030, 598.3837891, -777.6799316, 786.3875732
1: -292.4148254, 746.4031372, -287.3129883, 730.6085205, -1023.0233154, 1033.7160645
2: -204.1958008, 790.3706055, -199.7297058, 772.8367310, -977.0325317, 990.1003418
3: -521.0043945, 760.0877075, -510.0922241, 743.9570923, -1264.9614258, 1270.1799316
4: -325.3488159, 811.5093994, -317.8653870, 793.4354248, -1118.7841797, 1129.3747559

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2059717, upper bound: 843.2059730
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065728, upper bound: 843.2058853
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -179.2961731, 610.8264160, -184.1430817, 626.8763428, -806.1724854, 794.9694824
1: -292.4148254, 746.4031372, -301.7005615, 765.4571533, -1057.8719482, 1048.1037598
2: -204.1958008, 790.3706055, -209.4716339, 809.8760986, -1014.0718994, 999.8422241
3: -521.0043945, 760.0877075, -535.3128052, 779.7720947, -1300.7763672, 1295.4003906
4: -325.3488159, 811.5093994, -333.3037109, 831.6660767, -1157.0148926, 1144.8131104

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2059717, upper bound: 843.2059730
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065728, upper bound: 843.2058853
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -172.2730255, 587.1818848, -174.3998871, 593.9743042, -766.2473145, 761.5816650
1: -280.8097839, 717.5310059, -284.6991577, 725.7923584, -1006.6021729, 1002.2301636
2: -196.2395630, 759.6019897, -198.6237030, 767.6218262, -963.8613892, 958.2257080
3: -500.6431580, 730.5495605, -506.3637085, 739.0524292, -1239.6953125, 1236.9130859
4: -312.7213440, 779.9223633, -316.1923523, 788.3237915, -1101.0451660, 1096.1145020

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065609, upper bound: 843.2063138
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081950, upper bound: 843.2066628
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -172.2730255, 587.1818848, -182.8314972, 621.9671631, -794.2401123, 770.0133057
1: -280.8097839, 717.5310059, -298.8067932, 759.9968262, -1040.8066406, 1016.3377075
2: -196.2395630, 759.6019897, -208.1809540, 804.0384521, -1000.2780151, 967.7828979
3: -500.6431580, 730.5495605, -530.9179077, 774.1968384, -1274.8399658, 1261.4674072
4: -312.7213440, 779.9223633, -331.3768921, 825.8750000, -1138.5963135, 1111.2990723

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065609, upper bound: 843.2063138
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2081950, upper bound: 843.2066628
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -179.2961731, 610.8264160, -174.3998871, 593.9743042, -773.2704468, 785.2261963
1: -292.4148254, 746.4031372, -284.6991577, 725.7923584, -1018.2071533, 1031.1020508
2: -204.1958008, 790.3706055, -198.6237030, 767.6218262, -971.8176270, 988.9943237
3: -521.0043945, 760.0877075, -506.3637085, 739.0524292, -1260.0566406, 1266.4511719
4: -325.3488159, 811.5093994, -316.1923523, 788.3237915, -1113.6726074, 1127.7017822

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2059717, upper bound: 843.2062061
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067950, upper bound: 843.2059835
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -179.2961731, 610.8264160, -182.8314972, 621.9671631, -801.2632446, 793.6578369
1: -292.4148254, 746.4031372, -298.8067932, 759.9968262, -1052.4116211, 1045.2098389
2: -204.1958008, 790.3706055, -208.1809540, 804.0384521, -1008.2342529, 998.5514526
3: -521.0043945, 760.0877075, -530.9179077, 774.1968384, -1295.2011719, 1291.0054932
4: -325.3488159, 811.5093994, -331.3768921, 825.8750000, -1151.2238770, 1142.8862305

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2062253, upper bound: 843.2062061
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067950, upper bound: 843.2059835
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -173.9681702, 592.3180542, -175.5612030, 598.3837891, -772.3519287, 767.8792725
1: -283.2098389, 723.9351807, -287.3129883, 730.6085205, -1013.8182983, 1011.2481079
2: -198.2450714, 766.4570923, -199.7297058, 772.8367310, -971.0817871, 966.1867676
3: -505.2448425, 737.4182129, -510.0922241, 743.9570923, -1249.2019043, 1247.5104980
4: -315.9694824, 787.1150513, -317.8653870, 793.4354248, -1109.4049072, 1104.9804688

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071288, upper bound: 843.2065721
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2083064, upper bound: 843.2068566
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -173.9681702, 592.3180542, -184.1430817, 626.8763428, -800.8444824, 776.4611206
1: -283.2098389, 723.9351807, -301.7005615, 765.4571533, -1048.6669922, 1025.6357422
2: -198.2450714, 766.4570923, -209.4716339, 809.8760986, -1008.1211548, 975.9287109
3: -505.2448425, 737.4182129, -535.3128052, 779.7720947, -1285.0169678, 1272.7309570
4: -315.9694824, 787.1150513, -333.3037109, 831.6660767, -1147.6354980, 1120.4187012

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071288, upper bound: 843.2065721
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2083064, upper bound: 843.2068566
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -180.2173462, 613.6436768, -175.5612030, 598.3837891, -778.6011353, 789.2048950
1: -293.4556274, 749.9676514, -287.3129883, 730.6085205, -1024.0642090, 1037.2806396
2: -205.3023071, 794.1782837, -199.7297058, 772.8367310, -978.1389160, 993.9079590
3: -523.2786865, 763.9072266, -510.0922241, 743.9570923, -1267.2358398, 1273.9993896
4: -327.1666565, 815.4519653, -317.8653870, 793.4354248, -1120.6020508, 1133.3173828

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2068860, upper bound: 843.2064823
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070170, upper bound: 843.2062832
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -180.2173462, 613.6436768, -184.1430817, 626.8763428, -807.0936890, 797.7867432
1: -293.4556274, 749.9676514, -301.7005615, 765.4571533, -1058.9128418, 1051.6682129
2: -205.3023071, 794.1782837, -209.4716339, 809.8760986, -1015.1783447, 1003.6499023
3: -523.2786865, 763.9072266, -535.3128052, 779.7720947, -1303.0507812, 1299.2199707
4: -327.1666565, 815.4519653, -333.3037109, 831.6660767, -1158.8327637, 1148.7556152

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2068860, upper bound: 843.2064823
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070170, upper bound: 843.2062832
time: 0.86 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.61 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2091608, upper bound: 843.2083712
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2079872, upper bound: 843.2079872
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2091608, upper bound: 843.2092864
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2079872, upper bound: 843.2089284
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2083244
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2080793
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2092646
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2090248
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2091451, upper bound: 843.2075373
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2080793, upper bound: 843.2072034
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2091451, upper bound: 843.2090131
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2080793, upper bound: 843.2087032
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2075300
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2073002
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2077470, upper bound: 843.2090258
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2072034, upper bound: 843.2088000
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2095234, upper bound: 843.2088535
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2089284, upper bound: 843.2086440
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2095234, upper bound: 843.2097934
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2089284, upper bound: 843.2095897
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2089064
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2087497
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2098466
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2096914
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2095383, upper bound: 843.2080587
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2090248, upper bound: 843.2078651
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2095384, upper bound: 843.2094802
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2090248, upper bound: 843.2093649
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2081215
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2079665
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2087687, upper bound: 843.2081215
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2087032, upper bound: 843.2094663
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2063006, upper bound: 843.2061507
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2079693, upper bound: 843.2065698
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2063006, upper bound: 843.2061507
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2079693, upper bound: 843.2065698
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2059717, upper bound: 843.2059730
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2065728, upper bound: 843.2058853
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2059717, upper bound: 843.2059730
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2065728, upper bound: 843.2058853
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2065609, upper bound: 843.2063138
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2081950, upper bound: 843.2066628
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2065609, upper bound: 843.2063138
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2081950, upper bound: 843.2066628
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2059717, upper bound: 843.2062061
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2067950, upper bound: 843.2059835
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2062253, upper bound: 843.2062061
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2067950, upper bound: 843.2059835
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2071288, upper bound: 843.2065721
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2083064, upper bound: 843.2068566
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2071288, upper bound: 843.2065721
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2083064, upper bound: 843.2068566
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2068860, upper bound: 843.2064823
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2070170, upper bound: 843.2062832
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2068860, upper bound: 843.2064823
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.61
Output dim: 0, lower bound: -843.2070170, upper bound: 843.2062832
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2083972, upper bound: 843.2079269
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2069929, upper bound: 843.2080767
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2083972, upper bound: 843.2079269
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2083972, upper bound: 843.2080767
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2069929
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2069929
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2072469
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2069455, upper bound: 843.2072469
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2074459, upper bound: 843.2078875
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2074459, upper bound: 843.2078875
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2074822, upper bound: 843.2081832
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2074822, upper bound: 843.2081832
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2072911, upper bound: 843.2074592
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2072911, upper bound: 843.2074592
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2074125, upper bound: 843.2077203
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2074125, upper bound: 843.2077203
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2075455, upper bound: 843.2082961
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2075455, upper bound: 843.2082961
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2078378, upper bound: 843.2085774
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2078378, upper bound: 843.2085774
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2060440, upper bound: 843.2060440
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2060440, upper bound: 843.2060440
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2060440, upper bound: 843.2062406
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2060440, upper bound: 843.2062406
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2065075, upper bound: 843.2062903
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2065075, upper bound: 843.2063650
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2065075, upper bound: 843.2065372
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2065075, upper bound: 843.2066186
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2062903, upper bound: 843.2065075
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2062903, upper bound: 843.2065075
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2063650, upper bound: 843.2066656
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2063650, upper bound: 843.2067144
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2065902, upper bound: 843.2067773
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2065902, upper bound: 843.2068499
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2067781, upper bound: 843.2070111
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.61
Output dim: 0, lower bound: -843.2067781, upper bound: 843.2070937
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=901.5203857421875
rel_dist={0: [-843.2117464329822, 843.2117464329822]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2091795, upper bound: 843.2083798
time: 0.79 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2076610, upper bound: 843.2076610
time: 0.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.66 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.66
Output dim: 0, lower bound: -843.2091795, upper bound: 843.2083798
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.66
Output dim: 0, lower bound: -843.2076610, upper bound: 843.2076610

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -188.3924713, 641.2564087, -199.8674469, 680.4710693, -868.8635254, 841.1238403
1: -307.3268738, 783.6063843, -326.4258728, 831.2571411, -1138.5839844, 1110.0322266
2: -214.6040497, 829.6708374, -227.5879822, 880.0700684, -1094.6740723, 1057.2587891
3: -547.3501587, 798.2861328, -580.4912109, 846.7109985, -1394.0611572, 1378.7773438
4: -341.8905029, 851.8900757, -362.3172913, 903.3906250, -1245.2811279, 1214.2073975

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2084668, upper bound: 843.2077643
time: 0.75 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2083292, upper bound: 843.2070101
time: 0.97 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087060, upper bound: 843.2078122
time: 0.70 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -191.9384003, 653.1663818, -199.4010315, 678.5913086, -870.5297241, 852.5673828
1: -314.3370667, 797.7437744, -326.2332153, 828.9448242, -1143.2814941, 1123.9766846
2: -218.4419098, 843.8818970, -226.9938507, 877.1987915, -1095.6407471, 1070.8757324
3: -557.8200684, 812.7958374, -579.2211304, 844.4073486, -1402.2271729, 1392.0169678
4: -347.6372986, 866.6468506, -361.2475281, 900.5921631, -1248.2293701, 1227.8944092

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065973, upper bound: 843.2065028
time: 0.72 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070775, upper bound: 843.2070775
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.06 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 0, lower bound: -843.2083292, upper bound: 843.2070101
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 0, lower bound: -843.2087060, upper bound: 843.2078122
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 0, lower bound: -843.2065973, upper bound: 843.2065028
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 0, lower bound: -843.2070775, upper bound: 843.2070775

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -188.3924713, 641.2564087, -195.7923584, 666.6343994, -855.0268555, 837.0487671
1: -307.3268738, 783.6063843, -319.7748413, 814.4097900, -1121.7366943, 1103.3812256
2: -214.6040497, 829.6708374, -222.9266052, 862.2442627, -1076.8482666, 1052.5974121
3: -547.3501587, 798.2861328, -568.7651367, 829.4143677, -1376.7645264, 1367.0512695
4: -341.8905029, 851.8900757, -354.9081421, 885.1726685, -1227.0631104, 1206.7982178

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079251, upper bound: 843.2067605
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2079251, upper bound: 843.2070101
time: 0.86 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -187.7154694, 638.9248657, -196.0904236, 667.2966309, -855.0119019, 835.0152588
1: -306.1456909, 780.7756958, -319.6838074, 815.3701172, -1121.5153809, 1100.4594727
2: -213.8250275, 826.6771240, -223.3088074, 863.3056641, -1077.1307373, 1049.9859619
3: -545.3225708, 795.4010010, -569.0975342, 830.6007080, -1375.9233398, 1364.4984131
4: -340.6587830, 848.8321533, -355.5973816, 886.2728271, -1226.9315186, 1204.4291992

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2087060, upper bound: 843.2077355
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2084038, upper bound: 843.2076996
time: 0.80 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -191.9384003, 653.1663818, -195.7086945, 666.1958618, -858.1342773, 848.8750610
1: -314.3370667, 797.7437744, -320.1889954, 813.7379150, -1128.0747070, 1117.9326172
2: -218.4419098, 843.8818970, -222.7159882, 861.0163574, -1079.4582520, 1066.5977783
3: -557.8200684, 812.7958374, -568.5337524, 828.8266602, -1386.6464844, 1381.3295898
4: -347.6372986, 866.6468506, -354.4762878, 884.0972900, -1231.7346191, 1221.1231689

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061408, upper bound: 843.2061408
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061408, upper bound: 843.2065028
time: 0.75 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -191.0828552, 650.2592163, -195.0352020, 663.4647827, -854.5476074, 845.2944336
1: -312.8376465, 794.2415771, -318.4307251, 810.6621704, -1123.4997559, 1112.6721191
2: -217.4761505, 840.1472168, -222.1302948, 858.0791016, -1075.5549316, 1062.2774658
3: -555.2542114, 809.2076416, -566.2636108, 825.8588257, -1381.1127930, 1375.4711914
4: -346.1271362, 862.8562622, -353.5804138, 881.0428467, -1227.1696777, 1216.4366455

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067518, upper bound: 843.2067856
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069822, upper bound: 843.2069822
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.00 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -843.2079251, upper bound: 843.2067605
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -843.2079251, upper bound: 843.2070101
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -843.2087060, upper bound: 843.2077355
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -843.2084038, upper bound: 843.2076996
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -843.2061408, upper bound: 843.2061408
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -843.2061408, upper bound: 843.2065028
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -843.2067518, upper bound: 843.2067856
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 0, lower bound: -843.2069822, upper bound: 843.2069822

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -184.1930847, 626.9782715, -195.7923584, 666.6343994, -850.8273315, 822.7706299
1: -300.4424438, 766.2285767, -319.7748413, 814.4097900, -1114.8522949, 1086.0034180
2: -209.7961884, 811.2807007, -222.9266052, 862.2442627, -1072.0404053, 1034.2071533
3: -535.2235718, 780.4590454, -568.7651367, 829.4143677, -1364.6379395, 1349.2241211
4: -334.2570190, 833.0983276, -354.9081421, 885.1726685, -1219.4296875, 1188.0064697

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2064277, upper bound: 843.2053975
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067826, upper bound: 843.2064174
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070002, upper bound: 843.2064510
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -185.4147034, 630.6639404, -195.7923584, 666.6343994, -852.0490723, 826.4562988
1: -302.0009155, 770.8518677, -319.7748413, 814.4097900, -1116.4106445, 1090.6267090
2: -211.2521362, 816.2529297, -222.9266052, 862.2442627, -1073.4963379, 1039.1793213
3: -538.4077759, 785.4230957, -568.7651367, 829.4143677, -1367.8220215, 1354.1882324
4: -336.6280823, 838.2955933, -354.9081421, 885.1726685, -1221.8004150, 1193.2037354

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2064277, upper bound: 843.2060407
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067826, upper bound: 843.2067453
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070002, upper bound: 843.2068148
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -175.8406219, 599.2316895, -193.2030945, 657.7921143, -833.6326904, 792.4346313
1: -286.6120300, 732.2100220, -314.8589478, 803.7290039, -1090.3409424, 1047.0688477
2: -200.3218994, 775.1487427, -220.0100555, 850.9993896, -1051.3212891, 995.1587524
3: -510.8963318, 745.6528320, -560.6564331, 818.6066284, -1329.5029297, 1306.3093262
4: -319.2139587, 795.8314819, -350.3707886, 873.5561523, -1192.7701416, 1146.2021484

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080798, upper bound: 843.2072235
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2083153, upper bound: 843.2076631
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2083162, upper bound: 843.2076637
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -182.7118073, 622.4624634, -194.1128082, 660.7714844, -843.4832764, 816.5752563
1: -297.9226379, 760.5625000, -316.4385986, 807.3608398, -1105.2834473, 1077.0010986
2: -208.0994720, 805.3492432, -221.0470886, 854.8533936, -1062.9528809, 1026.3963623
3: -530.7606201, 774.6022949, -563.3426514, 822.3736572, -1353.1340332, 1337.9449463
4: -331.5535583, 826.7779541, -351.9961243, 877.5479126, -1209.1011963, 1178.7740479

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2080175, upper bound: 843.2071400
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2082442, upper bound: 843.2076218
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2082464, upper bound: 843.2076230
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -195.7086945, 666.1958618, -854.6710205, 837.0131226
1: -308.6676941, 783.1837769, -320.1889954, 813.7379150, -1122.4052734, 1103.3728027
2: -214.4147797, 828.5700073, -222.7159882, 861.0163574, -1075.4311523, 1051.2860107
3: -547.7684326, 797.9298096, -568.5337524, 828.8266602, -1376.5947266, 1366.4635010
4: -341.2081909, 850.9271851, -354.4762878, 884.0972900, -1225.3054199, 1205.4034424

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061408, upper bound: 843.2061408
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061408, upper bound: 843.2061408
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -195.7086945, 666.1958618, -853.3190308, 831.9498901
1: -305.6914673, 777.5458984, -320.1889954, 813.7379150, -1119.4291992, 1097.7348633
2: -213.0952911, 822.5731201, -222.7159882, 861.0163574, -1074.1116943, 1045.2889404
3: -543.3413086, 792.1798706, -568.5337524, 828.8266602, -1372.1679688, 1360.7136230
4: -339.2044983, 844.9370728, -354.4762878, 884.0972900, -1223.3017578, 1199.4133301

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061408, upper bound: 843.2065028
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061408, upper bound: 843.2065028
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -178.0934143, 606.9835815, -191.9297333, 653.2749634, -831.3681641, 798.9133301
1: -291.3469543, 741.2520752, -313.1987305, 798.1904907, -1089.5373535, 1054.4508057
2: -202.7034454, 783.9745483, -218.5806580, 844.8837280, -1047.5870361, 1002.5551758
3: -517.3498535, 754.8475342, -557.1419678, 812.9998169, -1330.3496094, 1311.9891357
4: -322.6315308, 804.9712524, -347.9551697, 867.3958130, -1190.0273438, 1152.9263916

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067518, upper bound: 843.2067856
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067518, upper bound: 843.2067856
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -186.8278656, 636.0616455, -193.1687622, 657.2799683, -844.1077881, 829.2304077
1: -306.0275574, 776.7873535, -315.3947754, 803.0733032, -1109.1007080, 1092.1820068
2: -212.6070099, 821.7566528, -219.9976196, 850.0734253, -1062.6802979, 1041.7542725
3: -543.0471191, 791.3373413, -560.8603516, 818.0722046, -1361.1192627, 1352.1976318
4: -338.3635254, 843.9187622, -350.1860962, 872.7912598, -1211.1546631, 1194.1048584

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069822, upper bound: 843.2069822
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069822, upper bound: 843.2069822
time: 0.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.39 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2067826, upper bound: 843.2064174
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2070002, upper bound: 843.2064510
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2067826, upper bound: 843.2067453
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2070002, upper bound: 843.2068148
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2083153, upper bound: 843.2076631
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2083162, upper bound: 843.2076637
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2082442, upper bound: 843.2076218
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2082464, upper bound: 843.2076230
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2061408, upper bound: 843.2061408
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2061408, upper bound: 843.2061408
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2061408, upper bound: 843.2065028
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2061408, upper bound: 843.2065028
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2067518, upper bound: 843.2067856
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2067518, upper bound: 843.2067856
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2069822, upper bound: 843.2069822
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -843.2069822, upper bound: 843.2069822

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -181.3740997, 617.7127686, -183.3619080, 625.0971069, -806.4711914, 801.0747070
1: -295.7040710, 754.8870850, -299.2607422, 763.5734863, -1059.2775879, 1054.1477051
2: -206.5743256, 799.2879028, -208.7689514, 808.2822876, -1014.8566284, 1008.0568848
3: -526.9607544, 768.7731323, -532.5759888, 777.2909546, -1304.2514648, 1301.3491211
4: -329.1546936, 820.6962891, -332.4024658, 829.6213989, -1158.7761230, 1153.0985107

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055819, upper bound: 843.2053504
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061674, upper bound: 843.2054799
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -182.2652130, 620.6273193, -190.8831177, 650.3696899, -832.6348877, 811.5102539
1: -297.2849426, 758.4268799, -311.7856140, 794.4739380, -1091.7586670, 1070.2122803
2: -207.5910034, 803.0576782, -217.3200836, 841.1900024, -1048.7807617, 1020.3777466
3: -529.6265259, 772.4389648, -554.5103760, 808.9580688, -1338.5844727, 1326.9493408
4: -330.7457886, 824.6017456, -345.9969482, 863.4609985, -1194.2066650, 1170.5986328

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063659, upper bound: 843.2051767
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065162, upper bound: 843.2057507
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -182.7092438, 621.7400513, -183.3619080, 625.0971069, -807.8063354, 805.1019287
1: -297.4724731, 759.9272461, -299.2607422, 763.5734863, -1061.0458984, 1059.1879883
2: -208.1600189, 804.7003784, -208.7689514, 808.2822876, -1016.4423218, 1013.4693604
3: -530.4882812, 774.1744995, -532.5759888, 777.2909546, -1307.7791748, 1306.7502441
4: -331.7312927, 826.3717041, -332.4024658, 829.6213989, -1161.3526611, 1158.7738037

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061980, upper bound: 843.2056871
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061674, upper bound: 843.2056613
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -183.3900909, 624.0405273, -190.8831177, 650.3696899, -833.7597046, 814.9234619
1: -298.6717224, 762.7234497, -311.7856140, 794.4739380, -1093.1456299, 1074.5089111
2: -208.9360046, 807.6621704, -217.3200836, 841.1900024, -1050.1258545, 1024.9822998
3: -532.5125122, 777.0484619, -554.5103760, 808.9580688, -1341.4703369, 1331.5588379
4: -332.9429016, 829.3998413, -345.9969482, 863.4609985, -1196.4038086, 1175.3967285

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070867, upper bound: 843.2055802
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2075889, upper bound: 843.2063155
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -171.9938660, 585.8408813, -183.2016907, 622.9725342, -794.9663696, 769.0425415
1: -280.3800354, 715.9238892, -298.6640930, 761.3792725, -1041.7590332, 1014.5878906
2: -195.9562225, 757.8064575, -208.6579132, 805.9089355, -1001.8651733, 966.4643555
3: -499.8118591, 729.1611938, -531.8270874, 775.7421265, -1275.5539551, 1260.9882812
4: -312.2877502, 778.1170044, -332.3575439, 827.4956665, -1139.7834473, 1110.4743652

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071789, upper bound: 843.2070921
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071789, upper bound: 843.2063211
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -173.7592468, 592.1125488, -191.6856079, 652.2725830, -826.0316772, 783.7981567
1: -283.2336121, 723.4853516, -312.9841919, 796.6494751, -1079.8828125, 1036.4694824
2: -197.9553375, 765.9430542, -218.1466064, 843.3302612, -1041.2856445, 984.0896606
3: -504.9021912, 736.7852173, -556.4667358, 811.5080566, -1316.4101562, 1293.2519531
4: -315.4473572, 786.4019775, -347.1580200, 866.0208130, -1181.4681396, 1133.5599365

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065552, upper bound: 843.2070910
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065552, upper bound: 843.2057949
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -178.9748383, 609.4190674, -184.2054901, 626.2609253, -805.2357788, 793.6243896
1: -291.8930664, 744.7033081, -300.4215393, 765.3898926, -1057.2829590, 1045.1248779
2: -203.8609161, 788.4602661, -209.8050842, 810.1649780, -1014.0258789, 998.2653809
3: -520.0203247, 758.5564575, -534.8085327, 779.8979492, -1299.9182129, 1293.3649902
4: -324.8274231, 809.5392456, -334.1566772, 831.9008179, -1156.7280273, 1143.6959229

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065540, upper bound: 843.2070486
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065540, upper bound: 843.2061101
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -180.4559021, 614.8317871, -192.6751099, 655.4826660, -835.9385376, 807.5068970
1: -294.2615967, 751.2042236, -314.7531433, 800.5606079, -1094.8222656, 1065.9569092
2: -205.5336456, 795.4698486, -219.2570190, 847.4486694, -1052.9822998, 1014.7268677
3: -524.2736206, 765.0591431, -559.4636230, 815.5718994, -1339.8454590, 1324.5224609
4: -327.4596863, 816.6406860, -348.9013367, 870.2966919, -1197.7563477, 1165.5419922

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065162, upper bound: 843.2070489
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065162, upper bound: 843.2076230
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -184.1930847, 626.9782715, -815.4533691, 825.4973755
1: -308.6676941, 783.1837769, -300.4424438, 766.2285767, -1074.8962402, 1083.6262207
2: -214.4147797, 828.5700073, -209.7961884, 811.2807007, -1025.6954346, 1038.3662109
3: -547.7684326, 797.9298096, -535.2235718, 780.4590454, -1328.2274170, 1333.1533203
4: -341.2081909, 850.9271851, -334.2570190, 833.0983276, -1174.3063965, 1185.1842041

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060304, upper bound: 843.2059242
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060597, upper bound: 843.2060597
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -188.4751282, 641.3044434, -188.4751282, 641.3044434, -829.7795410, 829.7795410
1: -308.6676941, 783.1837769, -308.6676941, 783.1837769, -1091.8514404, 1091.8514404
2: -214.4147797, 828.5700073, -214.4147797, 828.5700073, -1042.9847412, 1042.9847412
3: -547.7684326, 797.9298096, -547.7684326, 797.9298096, -1345.6978760, 1345.6978760
4: -341.2081909, 850.9271851, -341.2081909, 850.9271851, -1192.1352539, 1192.1352539

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060304, upper bound: 843.2059242
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060597, upper bound: 843.2060597
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -184.1930847, 626.9782715, -814.1014404, 820.4341431
1: -305.6914673, 777.5458984, -300.4424438, 766.2285767, -1071.9200439, 1077.9882812
2: -213.0952911, 822.5731201, -209.7961884, 811.2807007, -1024.3759766, 1032.3692627
3: -543.3413086, 792.1798706, -535.2235718, 780.4590454, -1323.8002930, 1327.4034424
4: -339.2044983, 844.9370728, -334.2570190, 833.0983276, -1172.3028564, 1179.1940918

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063539, upper bound: 843.2063297
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063539, upper bound: 843.2064576
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -187.1232147, 636.2412109, -188.4751282, 641.3044434, -828.4276123, 824.7163086
1: -305.6914673, 777.5458984, -308.6676941, 783.1837769, -1088.8752441, 1086.2135010
2: -213.0952911, 822.5731201, -214.4147797, 828.5700073, -1041.6652832, 1036.9879150
3: -543.3413086, 792.1798706, -547.7684326, 797.9298096, -1341.2711182, 1339.9479980
4: -339.2044983, 844.9370728, -341.2081909, 850.9271851, -1190.1317139, 1186.1448975

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063539, upper bound: 843.2063297
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063539, upper bound: 843.2064576
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -178.0934143, 606.9835815, -182.7092438, 621.7400513, -799.8332520, 789.6928101
1: -291.3469543, 741.2520752, -297.4724731, 759.9272461, -1051.2741699, 1038.7246094
2: -202.7034454, 783.9745483, -208.1600189, 804.7003784, -1007.4038086, 992.1345825
3: -517.3498535, 754.8475342, -530.4882812, 774.1744995, -1291.5242920, 1285.3358154
4: -322.6315308, 804.9712524, -331.7312927, 826.3717041, -1149.0031738, 1136.7025146

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060304, upper bound: 843.2062755
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060304, upper bound: 843.2067856
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -178.0934143, 606.9835815, -183.8621674, 625.5309448, -803.6240845, 790.8455811
1: -291.3469543, 741.2520752, -300.1998596, 764.4611816, -1055.8081055, 1041.4517822
2: -202.7034454, 783.9745483, -209.3811798, 808.7459717, -1011.4494019, 993.3557129
3: -517.3498535, 754.8475342, -533.8168335, 778.6887817, -1296.0385742, 1288.6643066
4: -322.6315308, 804.9712524, -333.3177490, 830.6459351, -1153.2773438, 1138.2889404

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060304, upper bound: 843.2062755
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060304, upper bound: 843.2067856
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -186.8278656, 636.0616455, -183.3900909, 624.0405273, -810.8683472, 819.4517212
1: -306.0275574, 776.7873535, -298.6717224, 762.7234497, -1068.7509766, 1075.4591064
2: -212.6070099, 821.7566528, -208.9360046, 807.6621704, -1020.2691650, 1030.6925049
3: -543.0471191, 791.3373413, -532.5125122, 777.0484619, -1320.0955811, 1323.8496094
4: -338.3635254, 843.9187622, -332.9429016, 829.3998413, -1167.7631836, 1176.8615723

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060597, upper bound: 843.2065000
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060597, upper bound: 843.2069822
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -186.8278656, 636.0616455, -185.4421082, 630.6641846, -817.4920044, 821.5037231
1: -306.0275574, 776.7873535, -302.9889221, 770.6894531, -1076.7169189, 1079.7762451
2: -212.6070099, 821.7566528, -211.1721344, 815.3331299, -1027.9400635, 1032.9287109
3: -543.0471191, 791.3373413, -538.4769897, 785.1505127, -1328.1976318, 1329.8143311
4: -338.3635254, 843.9187622, -336.1414795, 837.4859009, -1175.8493652, 1180.0603027

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060597, upper bound: 843.2065000
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2060597, upper bound: 843.2069822
time: 1.09 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.46 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2055819, upper bound: 843.2053504
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2061674, upper bound: 843.2054799
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2063659, upper bound: 843.2051767
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2065162, upper bound: 843.2057507
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2061980, upper bound: 843.2056871
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2061674, upper bound: 843.2056613
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2070867, upper bound: 843.2055802
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2075889, upper bound: 843.2063155
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2071789, upper bound: 843.2070921
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2071789, upper bound: 843.2063211
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2065552, upper bound: 843.2070910
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2065552, upper bound: 843.2057949
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2065540, upper bound: 843.2070486
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2065540, upper bound: 843.2061101
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2065162, upper bound: 843.2070489
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2065162, upper bound: 843.2076230
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2060304, upper bound: 843.2059242
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2060597, upper bound: 843.2060597
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2060304, upper bound: 843.2059242
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2060597, upper bound: 843.2060597
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2063539, upper bound: 843.2063297
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2063539, upper bound: 843.2064576
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2063539, upper bound: 843.2063297
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2063539, upper bound: 843.2064576
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2060304, upper bound: 843.2062755
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2060304, upper bound: 843.2067856
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2060304, upper bound: 843.2062755
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2060304, upper bound: 843.2067856
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2060597, upper bound: 843.2065000
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2060597, upper bound: 843.2069822
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2060597, upper bound: 843.2065000
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.46
Output dim: 0, lower bound: -843.2060597, upper bound: 843.2069822

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -170.8866882, 581.7937622, -179.0872650, 610.4452515, -781.3319092, 760.8809814
1: -278.8908386, 710.8239136, -292.3978271, 745.6035767, -1024.4943848, 1003.2217407
2: -194.7112427, 752.6896362, -203.9328766, 789.2809448, -983.9921875, 956.6224976
3: -496.9226379, 724.1068726, -520.3251953, 759.0700073, -1255.9926758, 1244.4321289
4: -310.1272278, 773.1140137, -324.6457825, 810.2169189, -1120.3441162, 1097.7596436

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055819, upper bound: 843.2053504
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055819, upper bound: 843.2053504
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -180.8722229, 616.0836182, -183.0765228, 624.1182861, -804.9904785, 799.1601562
1: -294.8542175, 752.8505859, -298.8020020, 762.3748779, -1057.2291260, 1051.6525879
2: -205.9721222, 797.1860962, -208.4451752, 807.0206299, -1012.9927368, 1005.6312866
3: -525.4558105, 766.6262817, -531.7628784, 776.0769043, -1301.5325928, 1298.3891602
4: -328.1828918, 818.4721680, -331.8857422, 828.3383789, -1156.5212402, 1150.3579102

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061674, upper bound: 843.2054799
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054247, upper bound: 843.2054799
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -172.5794525, 586.8292236, -187.0538940, 637.0322876, -809.6116943, 773.8829956
1: -281.6849976, 717.3369141, -305.6441040, 778.2559204, -1059.9409180, 1022.9810181
2: -196.6121979, 759.2854614, -212.9803925, 823.9021606, -1020.5143433, 972.2658081
3: -501.8131409, 730.8677979, -543.5114746, 792.5419312, -1294.3548584, 1274.3791504
4: -313.3210449, 779.9204712, -339.1030273, 845.8054199, -1159.1263428, 1119.0233154

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063659, upper bound: 843.2051767
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2063659, upper bound: 843.2051767
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -181.6207275, 617.9006348, -188.6033478, 642.6851807, -824.3059082, 806.5039673
1: -296.9969177, 754.5702515, -308.0455933, 785.0435181, -1082.0400391, 1062.6158447
2: -206.6700134, 798.7661743, -214.7224274, 831.2492065, -1037.9191895, 1013.4885864
3: -528.0310669, 768.7924805, -547.9324951, 799.3187866, -1327.3494873, 1316.7247314
4: -328.9238281, 820.5295410, -341.8619995, 853.2484741, -1182.1722412, 1162.3913574

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2057679, upper bound: 843.2050688
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065162, upper bound: 843.2057507
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065162, upper bound: 843.2057507
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -172.0807343, 585.3051147, -179.0872650, 610.4452515, -782.5260010, 764.3923950
1: -280.3955688, 715.2485962, -292.3978271, 745.6035767, -1025.9991455, 1007.6464233
2: -196.1299896, 757.4588013, -203.9328766, 789.2809448, -985.4109497, 961.3916626
3: -500.0113831, 728.8901978, -520.3251953, 759.0700073, -1259.0814209, 1249.2153320
4: -312.4447327, 778.1140747, -324.6457825, 810.2169189, -1122.6616211, 1102.7598877

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061980, upper bound: 843.2056871
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061980, upper bound: 843.2056871
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -182.3417511, 620.5905762, -183.0765228, 624.1182861, -806.4600220, 803.6671143
1: -296.8294067, 758.4749146, -298.8020020, 762.3748779, -1059.2043457, 1057.2768555
2: -207.7109833, 803.2200928, -208.4451752, 807.0206299, -1014.7316284, 1011.6652222
3: -529.3607788, 772.6306763, -531.7628784, 776.0769043, -1305.4375000, 1304.3933105
4: -331.0033569, 824.7728271, -331.8857422, 828.3383789, -1159.3416748, 1156.6585693

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2064470, upper bound: 843.2056613
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2064470, upper bound: 843.2056613
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -173.4450836, 589.3643188, -187.0538940, 637.0322876, -810.4773560, 776.4180908
1: -282.5744324, 720.5694580, -305.6441040, 778.2559204, -1060.8303223, 1026.2136230
2: -197.6498566, 762.7757568, -212.9803925, 823.9021606, -1021.5519409, 975.7561035
3: -503.9001770, 734.4107056, -543.5114746, 792.5419312, -1296.4420166, 1277.9221191
4: -315.0426331, 783.5739746, -339.1030273, 845.8054199, -1160.8480225, 1122.6768799

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070867, upper bound: 843.2055802
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070867, upper bound: 843.2055802
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -180.5475006, 614.0549316, -188.6033478, 642.6851807, -823.2326660, 802.6582031
1: -294.5497742, 750.2497559, -308.0455933, 785.0435181, -1079.5932617, 1058.2954102
2: -205.6331635, 794.3887329, -214.7224274, 831.2492065, -1036.8823242, 1009.1111450
3: -524.4255981, 764.4099121, -547.9324951, 799.3187866, -1323.7441406, 1312.3424072
4: -327.4491272, 815.9871216, -341.8619995, 853.2484741, -1180.6976318, 1157.8491211

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2072547, upper bound: 843.2057394
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2075889, upper bound: 843.2063155
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2075889, upper bound: 843.2063155
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -168.4867554, 573.9703369, -183.2016907, 622.9725342, -791.4591675, 757.1718750
1: -274.6933594, 701.4699707, -298.6640930, 761.3792725, -1036.0726318, 1000.1340332
2: -191.9448547, 742.4895630, -208.6579132, 805.9089355, -997.8537598, 951.1474609
3: -489.7416992, 714.2951660, -531.8270874, 775.7421265, -1265.4836426, 1246.1223145
4: -305.9037476, 762.4496460, -332.3575439, 827.4956665, -1133.3994141, 1094.8071289

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2053202, upper bound: 843.2052490
time: 1.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071789, upper bound: 843.2070921
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071789, upper bound: 843.2070921
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -170.0892944, 578.8132324, -183.2016907, 622.9725342, -793.0618286, 762.0148926
1: -276.9266052, 707.5168457, -298.6640930, 761.3792725, -1038.3057861, 1006.1809082
2: -193.8440857, 748.9759521, -208.6579132, 805.9089355, -999.7530518, 957.6338501
3: -494.0787659, 720.7940674, -531.8270874, 775.7421265, -1269.8209229, 1252.6210938
4: -308.9916992, 769.2609253, -332.3575439, 827.4956665, -1136.4871826, 1101.6179199

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2053202, upper bound: 843.2057809
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071789, upper bound: 843.2076631
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2071789, upper bound: 843.2076631
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -170.1440430, 579.9107666, -191.6856079, 652.2725830, -822.4165649, 771.5963745
1: -277.3519897, 708.6238403, -312.9841919, 796.6494751, -1074.0014648, 1021.6079102
2: -193.8186340, 750.2022705, -218.1466064, 843.3302612, -1037.1489258, 968.3488770
3: -494.5145874, 721.4920654, -556.4667358, 811.5080566, -1306.0222168, 1277.9587402
4: -308.8651733, 770.2965088, -347.1580200, 866.0208130, -1174.8859863, 1117.4545898

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061146, upper bound: 843.2063950
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065552, upper bound: 843.2070910
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065552, upper bound: 843.2070910
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -171.9152985, 585.2678223, -191.6856079, 652.2725830, -824.1878662, 776.9533691
1: -279.8724060, 715.2883301, -312.9841919, 796.6494751, -1076.5217285, 1028.2724609
2: -195.9095154, 757.3363037, -218.1466064, 843.3302612, -1039.2397461, 975.4829102
3: -499.3131409, 728.6403809, -556.4667358, 811.5080566, -1310.8208008, 1285.1071777
4: -312.2446899, 777.7818604, -347.1580200, 866.0208130, -1178.2653809, 1124.9399414

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061146, upper bound: 843.2051448
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065552, upper bound: 843.2057949
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065552, upper bound: 843.2076637
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -175.5968323, 597.9061890, -184.2054901, 626.2609253, -801.8577881, 782.1114502
1: -286.4527893, 730.6997681, -300.4215393, 765.3898926, -1051.8426514, 1031.1213379
2: -200.0019226, 773.6372070, -209.8050842, 810.1649780, -1010.1668701, 983.4422607
3: -510.3780518, 744.2017212, -534.8085327, 779.8979492, -1290.2760010, 1279.0102539
4: -318.6940918, 794.4288940, -334.1566772, 831.9008179, -1150.5948486, 1128.5855713

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2051994, upper bound: 843.2043345
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065540, upper bound: 843.2070486
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065540, upper bound: 843.2070486
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -176.4324493, 600.4392700, -184.2054901, 626.2609253, -802.6933594, 784.6445923
1: -287.3247070, 733.9182739, -300.4215393, 765.3898926, -1052.7145996, 1034.3398438
2: -201.0060730, 777.0867920, -209.8050842, 810.1649780, -1011.1709595, 986.8918457
3: -512.3833618, 747.6724243, -534.8085327, 779.8979492, -1292.2812500, 1282.4808350
4: -320.3539734, 798.0011597, -334.1566772, 831.9008179, -1152.2547607, 1132.1575928

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2051994, upper bound: 843.2057809
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065540, upper bound: 843.2061101
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065540, upper bound: 843.2076218
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -176.9682007, 602.9567261, -192.6751099, 655.4826660, -832.4508667, 795.6317749
1: -288.6310730, 736.7567749, -314.7531433, 800.5606079, -1089.1916504, 1051.5097656
2: -201.5466614, 780.1802979, -219.2570190, 847.4486694, -1048.9953613, 999.4373169
3: -514.3063965, 750.2458496, -559.4636230, 815.5718994, -1329.8780518, 1309.7094727
4: -321.1245422, 801.0511475, -348.9013367, 870.2966919, -1191.4211426, 1149.9523926

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2057679, upper bound: 843.2063185
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065162, upper bound: 843.2070489
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065162, upper bound: 843.2070489
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -177.9957886, 606.0978394, -192.6751099, 655.4826660, -833.4784546, 798.7728882
1: -289.8466187, 740.7112427, -314.7531433, 800.5606079, -1090.4072266, 1055.4637451
2: -202.7730408, 784.4100342, -219.2570190, 847.4486694, -1050.2216797, 1003.6670532
3: -516.8694458, 754.4748535, -559.4636230, 815.5718994, -1332.4412842, 1313.9383545
4: -323.1336365, 805.4294434, -348.9013367, 870.2966919, -1193.4302979, 1154.3308105

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2057679, upper bound: 843.2069978
time: 1.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065162, upper bound: 843.2076230
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2065162, upper bound: 843.2076230
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -175.5612030, 598.3837891, -181.3740997, 617.7127686, -793.2739868, 779.7578735
1: -287.3129883, 730.6085205, -295.7040710, 754.8870850, -1042.2000732, 1026.3126221
2: -199.7297058, 772.8367310, -206.5743256, 799.2879028, -999.0175781, 979.4110718
3: -510.0922241, 743.9570923, -526.9607544, 768.7731323, -1278.8653564, 1270.9178467
4: -317.8653870, 793.4354248, -329.1546936, 820.6962891, -1138.5616455, 1122.5899658

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2053504, upper bound: 843.2055819
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054799, upper bound: 843.2061674
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -184.1430817, 626.8763428, -182.2652130, 620.6273193, -804.7703857, 809.1415405
1: -301.7005615, 765.4571533, -297.2849426, 758.4268799, -1060.1274414, 1062.7420654
2: -209.4716339, 809.8760986, -207.5910034, 803.0576782, -1012.5292969, 1017.4671021
3: -535.3128052, 779.7720947, -529.6265259, 772.4389648, -1307.7517090, 1309.3986816
4: -333.3037109, 831.6660767, -330.7457886, 824.6017456, -1157.9055176, 1162.4118652

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2051767, upper bound: 843.2063659
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2057507, upper bound: 843.2065162
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -175.5612030, 598.3837891, -185.2375641, 630.6828003, -806.2440186, 783.6213379
1: -287.3129883, 730.6085205, -303.1951904, 770.1984253, -1057.5113525, 1033.8037109
2: -199.7297058, 772.8367310, -210.7275085, 814.8322754, -1014.5620117, 983.5641479
3: -510.0922241, 743.9570923, -538.2476196, 784.5410156, -1294.6333008, 1282.2047119
4: -317.8653870, 793.4354248, -335.3729553, 836.7470093, -1154.6124268, 1128.8083496

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055426, upper bound: 843.2054235
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2053050, upper bound: 843.2050676
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -184.1430817, 626.8763428, -186.7439880, 635.5515747, -819.6946411, 813.6203613
1: -301.7005615, 765.4571533, -305.8703003, 776.1262207, -1077.8267822, 1071.3273926
2: -209.4716339, 809.8760986, -212.4390564, 821.1153564, -1030.5870361, 1022.3151245
3: -535.3128052, 779.7720947, -542.7861328, 790.6988525, -1326.0117188, 1322.5582275
4: -333.3037109, 831.6660767, -338.0534363, 843.2479858, -1176.5517578, 1169.7194824

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2052965, upper bound: 843.2054123
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2050773, upper bound: 843.2050773
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -174.3998871, 593.9743042, -181.3740997, 617.7127686, -792.1126709, 775.3483887
1: -284.6991577, 725.7923584, -295.7040710, 754.8870850, -1039.5859375, 1021.4962769
2: -198.6237030, 767.6218262, -206.5743256, 799.2879028, -997.9116211, 974.1961670
3: -506.3637085, 739.0524292, -526.9607544, 768.7731323, -1275.1368408, 1266.0130615
4: -316.1923523, 788.3237915, -329.1546936, 820.6962891, -1136.8886719, 1117.4783936

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056565, upper bound: 843.2058503
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056323, upper bound: 843.2063096
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -182.8314972, 621.9671631, -182.2652130, 620.6273193, -803.4588013, 804.2322388
1: -298.8067932, 759.9968262, -297.2849426, 758.4268799, -1057.2336426, 1057.2817383
2: -208.1809540, 804.0384521, -207.5910034, 803.0576782, -1011.2386475, 1011.6294556
3: -530.9179077, 774.1968384, -529.6265259, 772.4389648, -1303.3569336, 1303.8233643
4: -331.3768921, 825.8750000, -330.7457886, 824.6017456, -1155.9786377, 1156.6208496

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2064865, upper bound: 843.2069759
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2070489, upper bound: 843.2071856
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -174.3998871, 593.9743042, -185.2375641, 630.6828003, -805.0826416, 779.2118530
1: -284.6991577, 725.7923584, -303.1951904, 770.1984253, -1054.8972168, 1028.9874268
2: -198.6237030, 767.6218262, -210.7275085, 814.8322754, -1013.4559937, 978.3493042
3: -506.3637085, 739.0524292, -538.2476196, 784.5410156, -1290.9047852, 1277.3000488
4: -316.1923523, 788.3237915, -335.3729553, 836.7470093, -1152.9393311, 1123.6965332

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055096, upper bound: 843.2054945
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2052642, upper bound: 843.2051030
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -182.8314972, 621.9671631, -186.7439880, 635.5515747, -818.3830566, 808.7111816
1: -298.8067932, 759.9968262, -305.8703003, 776.1262207, -1074.9329834, 1065.8671875
2: -208.1809540, 804.0384521, -212.4390564, 821.1153564, -1029.2962646, 1016.4774780
3: -530.9179077, 774.1968384, -542.7861328, 790.6988525, -1321.6166992, 1316.9829102
4: -331.3768921, 825.8750000, -338.0534363, 843.2479858, -1174.6248779, 1163.9284668

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061003, upper bound: 843.2061239
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2052646, upper bound: 843.2054769
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2050426, upper bound: 843.2050992
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -175.5612030, 598.3837891, -182.7092438, 621.7400513, -797.3012695, 781.0930176
1: -287.3129883, 730.6085205, -297.4724731, 759.9272461, -1047.2402344, 1028.0810547
2: -199.7297058, 772.8367310, -208.1600189, 804.7003784, -1004.4300537, 980.9967041
3: -510.0922241, 743.9570923, -530.4882812, 774.1744995, -1284.2667236, 1274.4453125
4: -317.8653870, 793.4354248, -331.7312927, 826.3717041, -1144.2370605, 1125.1665039

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=901.5203857421875
rel_dist={0: [-843.2116141859327, 843.2116141859326]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1110.83 seconds
