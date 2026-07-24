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
execution time: IAR + LP analysis = 2.02 + 1.96 = 3.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -843.2117555, upper bound: 843.2117555


# Binary Search by BASE starts (time budget: 1196.02 seconds, max iter: 100)

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
rel_dist={0: [-843.2107600097642, 843.210760002115]}

## Binary Search Result
Binary search time: 82.65 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1113.37 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -843.1912192, upper bound: 843.1932989
time: 0.76 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2117435, upper bound: 843.2117435
time: 0.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.78 seconds
IS_B1, status: Status.VERIFIED, split count: 1, time: 1.78
Output dim: 0, lower bound: -843.1912192, upper bound: 843.1932989
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 0, lower bound: -843.2117435, upper bound: 843.2117435

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -204.6462860, 696.6473389, -901.3449707, 901.4689941
1: -334.7795410, 851.1083374, -334.6973877, 850.8936157, -1185.6730957, 1185.8056641
2: -233.0683441, 900.9995117, -233.0101318, 900.7739868, -1133.8422852, 1134.0096436
3: -594.6608887, 866.9309082, -594.5149536, 866.7125244, -1461.3732910, 1461.4458008
4: -370.8992920, 924.7813721, -370.8064880, 924.5512695, -1295.4505615, 1295.5878906

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069197, upper bound: 843.2102971
time: 1.25 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2058473, upper bound: 843.2058473
time: 0.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.01 seconds
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 0, lower bound: -843.2069197, upper bound: 843.2102971
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 0, lower bound: -843.2058473, upper bound: 843.2058473

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -195.9719086, 666.4939575, -871.1915283, 892.7947388
1: -334.7795410, 851.1083374, -320.6729126, 814.2442627, -1149.0235596, 1171.7812500
2: -233.0683441, 900.9995117, -223.1084137, 861.7239990, -1094.7923584, 1124.1079102
3: -594.6608887, 866.9309082, -569.2477417, 829.5389404, -1424.1998291, 1436.1787109
4: -370.8992920, 924.7813721, -355.0805054, 884.4744263, -1255.3736572, 1279.8618164

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2058455, upper bound: 843.2096694
time: 1.16 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043420, upper bound: 843.2046954
time: 0.77 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -204.3803406, 695.7320557, -216.6208496, 736.7177124, -941.0980225, 912.3527832
1: -334.2254639, 849.7663574, -353.7630310, 899.6384277, -1233.8638916, 1203.5294189
2: -232.7091522, 899.6076660, -246.6053619, 952.3645020, -1185.0736084, 1146.2128906
3: -593.7491455, 865.5722046, -628.6013794, 917.2362061, -1510.9853516, 1494.1734619
4: -370.3317871, 923.3668213, -392.5297241, 977.7860718, -1348.1179199, 1315.8964844

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_B2_B1

### Relational analysis result of IS_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047731, upper bound: 843.2053794
time: 0.98 seconds

## Relational analysis of IS_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043052, upper bound: 843.2043052
time: 0.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.85 seconds
IS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.85
Output dim: 0, lower bound: -843.2058455, upper bound: 843.2096694
IS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.85
Output dim: 0, lower bound: -843.2043420, upper bound: 843.2046954
IS_B2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.85
Output dim: 0, lower bound: -843.2047731, upper bound: 843.2053794
IS_B2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.85
Output dim: 0, lower bound: -843.2043052, upper bound: 843.2043052

## BFS IS instance: IS_B2_B1_B1

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -190.5143890, 647.8603516, -852.5578613, 887.3370972
1: -334.7795410, 851.1083374, -311.9286804, 791.4840698, -1126.2630615, 1163.0369873
2: -233.0683441, 900.9995117, -216.8975220, 837.4561768, -1070.5245361, 1117.8968506
3: -594.6608887, 866.9309082, -553.4931030, 806.4064941, -1401.0673828, 1420.4240723
4: -370.8992920, 924.7813721, -345.2073975, 859.6048584, -1230.5041504, 1269.9886475

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056816, upper bound: 843.2096253
time: 0.75 seconds

## Relational analysis of IS_B2_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2051159, upper bound: 843.2063308
time: 0.79 seconds

## BFS IS instance: IS_B2_B1_B2

### Backsubstitution after applying IS history:
0: -204.3977966, 695.8164673, -214.9246216, 731.8286743, -936.2264404, 910.7409668
1: -334.2704773, 849.8805542, -353.0226135, 893.8590698, -1228.1295166, 1202.9030762
2: -232.7335968, 899.7196655, -244.6210480, 945.3122559, -1178.0455322, 1144.3405762
3: -593.7982788, 865.6848755, -624.3003540, 910.8497314, -1504.6479492, 1489.9852295
4: -370.3714294, 923.4755249, -388.9274597, 970.1460571, -1340.5173340, 1312.4027100

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_B2_B1

### Relational analysis result of IS_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043420, upper bound: 843.2046954
time: 0.67 seconds

## Relational analysis of IS_B2_B1_B2_B2

### Relational analysis result of IS_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043380, upper bound: 843.2044572
time: 0.93 seconds

## BFS IS instance: IS_B2_B2_B1

### Backsubstitution after applying IS history:
0: -204.3803406, 695.7320557, -211.3331451, 718.5986328, -922.9790039, 907.0651245
1: -334.2254639, 849.7663574, -345.2428284, 877.5130005, -1211.7380371, 1195.0091553
2: -232.7091522, 899.6076660, -240.5852509, 928.7516479, -1161.4608154, 1140.1928711
3: -593.7491455, 865.5722046, -613.2625732, 894.7688599, -1488.5178223, 1478.8345947
4: -370.3317871, 923.3668213, -382.9114990, 953.5756836, -1323.9074707, 1306.2783203

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_B1

### Relational analysis result of IS_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022765, upper bound: 843.2009625
time: 1.01 seconds

## Relational analysis of IS_B2_B2_B1_B2

### Relational analysis result of IS_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047731, upper bound: 843.2053794
time: 0.82 seconds

## BFS IS instance: IS_B2_B2_B2

### Backsubstitution after applying IS history:
0: -204.0812836, 694.7272949, -222.8520660, 757.3113403, -961.3926392, 917.5792847
1: -333.7183533, 848.5408936, -364.4555054, 924.6558228, -1258.3740234, 1212.9963379
2: -232.3754578, 898.3300171, -253.5426941, 978.6007080, -1210.9761963, 1151.8726807
3: -592.8891602, 864.3291626, -645.8093262, 942.6522217, -1535.5413818, 1510.1384277
4: -369.8052368, 922.0638428, -403.2563477, 1004.6666260, -1374.4719238, 1325.3198242

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B2_B1

### Relational analysis result of IS_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2038563, upper bound: 843.2030544
time: 1.20 seconds

## Relational analysis of IS_B2_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043052, upper bound: 843.2043052
time: 0.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.04 seconds
IS_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -843.2056816, upper bound: 843.2096253
IS_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -843.2051159, upper bound: 843.2063308
IS_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -843.2043420, upper bound: 843.2046954
IS_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -843.2043380, upper bound: 843.2044572
IS_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -843.2022765, upper bound: 843.2009625
IS_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -843.2047731, upper bound: 843.2053794
IS_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -843.2038563, upper bound: 843.2030544
IS_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.04
Output dim: 0, lower bound: -843.2043052, upper bound: 843.2043052

## BFS IS instance: IS_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -182.7452393, 621.1705933, -825.8681641, 879.5679321
1: -334.7795410, 851.1083374, -299.4613342, 758.8191528, -1093.5983887, 1150.5697021
2: -233.0683441, 900.9995117, -208.0237579, 802.6618042, -1035.7301025, 1109.0233154
3: -594.6608887, 866.9309082, -530.9902954, 773.3157349, -1367.9765625, 1397.9208984
4: -370.8992920, 924.7813721, -331.0804749, 824.0029297, -1194.9022217, 1255.8618164

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055856, upper bound: 843.2096253
time: 1.06 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056816, upper bound: 843.2091459
time: 0.84 seconds

## BFS IS instance: IS_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -204.4359436, 695.9414062, -200.6152496, 681.8596191, -886.2955322, 896.5563965
1: -334.3552856, 850.0263062, -329.4119568, 832.5629883, -1166.9179688, 1179.4382324
2: -232.7866821, 899.8817139, -228.5471344, 881.2005615, -1113.9871826, 1128.4285889
3: -593.9490356, 865.8302002, -583.4202881, 848.5297241, -1442.4786377, 1449.2504883
4: -370.4492188, 923.6339111, -363.5925598, 904.5921021, -1275.0412598, 1287.2264404

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_B2_B1

### Relational analysis result of IS_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2051141, upper bound: 843.2062918
time: 0.89 seconds

## Relational analysis of IS_B2_B1_B1_B2_B2

### Relational analysis result of IS_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2045809, upper bound: 843.2048724
time: 0.74 seconds

## BFS IS instance: IS_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -204.3977966, 695.8164673, -208.2344666, 709.2060547, -913.6038818, 904.0509033
1: -334.2704773, 849.8805542, -342.3541565, 866.0283203, -1200.2988281, 1192.2346191
2: -232.7335968, 899.7196655, -236.9849396, 915.6755981, -1148.4091797, 1136.7044678
3: -593.7982788, 865.6848755, -604.9898682, 882.4639893, -1476.2622070, 1470.6748047
4: -370.3714294, 923.4755249, -376.6349182, 939.6113281, -1309.9827881, 1300.1103516

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B2_B1_A1

### Relational analysis result of IS_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2041992, upper bound: 843.2045982
time: 0.84 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2

### Relational analysis result of IS_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043420, upper bound: 843.2046954
time: 0.74 seconds

## BFS IS instance: IS_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -204.1402130, 694.9329224, -219.0049133, 744.0256958, -948.1658325, 913.9377441
1: -333.8447266, 848.7954712, -360.0397644, 908.7313232, -1242.5760498, 1208.8352051
2: -232.4510498, 898.5988159, -249.2545624, 961.1817627, -1193.6328125, 1147.8533936
3: -593.0841675, 864.5813599, -635.7150269, 925.9451294, -1519.0291748, 1500.2962646
4: -369.9197693, 922.3253784, -396.3262024, 986.1468506, -1356.0666504, 1318.6513672

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B2_B2_B1

### Relational analysis result of IS_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2039852, upper bound: 843.2041625
time: 0.78 seconds

## Relational analysis of IS_B2_B1_B2_B2_B2

### Relational analysis result of IS_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043380, upper bound: 843.2044572
time: 0.77 seconds

## BFS IS instance: IS_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -204.3803406, 695.7320557, -203.1809387, 690.2886353, -894.6689453, 898.9129639
1: -334.2254639, 849.7663574, -332.0028381, 842.9625244, -1177.1877441, 1181.7691650
2: -232.7091522, 899.6076660, -231.2542572, 891.9806519, -1124.6898193, 1130.8616943
3: -593.7491455, 865.5722046, -589.4570923, 859.7695923, -1453.5187988, 1455.0290527
4: -370.3317871, 923.3668213, -367.9657593, 915.9065552, -1286.2382812, 1291.3323975

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B1_B1_B1

### Relational analysis result of IS_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022760, upper bound: 843.2009625
time: 0.77 seconds

## Relational analysis of IS_B2_B2_B1_B1_B2

### Relational analysis result of IS_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022765, upper bound: 843.2007391
time: 0.85 seconds

## BFS IS instance: IS_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -204.1289825, 694.8851318, -221.6524963, 754.2523804, -958.3813477, 916.5374756
1: -333.8211365, 848.7246094, -363.1070557, 920.7637329, -1254.5848389, 1211.8315430
2: -232.4389191, 898.5305786, -252.5341339, 974.3565674, -1206.7955322, 1151.0644531
3: -593.0671387, 864.5166016, -643.7025146, 939.1434937, -1532.2106934, 1508.2191162
4: -369.8994141, 922.2689819, -401.8738403, 1000.2042847, -1370.1037598, 1324.1428223

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B1_B2_B1

### Relational analysis result of IS_B2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047679, upper bound: 843.2053794
time: 0.67 seconds

## Relational analysis of IS_B2_B2_B1_B2_B2

### Relational analysis result of IS_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2044335, upper bound: 843.2044427
time: 0.80 seconds

## BFS IS instance: IS_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -204.0812836, 694.7272949, -214.8024750, 729.3087769, -933.3900757, 909.5297241
1: -333.7183533, 848.5408936, -351.4609070, 890.5057983, -1224.2241211, 1200.0015869
2: -232.3754578, 898.3300171, -244.3223114, 942.1585693, -1174.5340576, 1142.6523438
3: -592.8891602, 864.3291626, -622.3866577, 908.0667114, -1500.9558105, 1486.7152100
4: -369.8052368, 922.0638428, -388.4810791, 967.4044189, -1337.2097168, 1310.5449219

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B2_B1_A1

### Relational analysis result of IS_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2037136, upper bound: 843.2029572
time: 0.93 seconds

## Relational analysis of IS_B2_B2_B2_B1_A2

### Relational analysis result of IS_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2038563, upper bound: 843.2030544
time: 0.72 seconds

## BFS IS instance: IS_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -203.8349304, 693.8806152, -230.0985565, 781.9533081, -985.7881470, 923.9790649
1: -333.3126221, 847.4983521, -377.4009399, 954.5874634, -1287.9000244, 1224.8992920
2: -232.1045227, 897.2526245, -262.0014038, 1010.2806396, -1242.3851318, 1159.2539062
3: -592.2056274, 863.2735596, -667.8724365, 973.3690186, -1565.5747070, 1531.1458740
4: -369.3724060, 920.9666748, -416.6943665, 1037.1672363, -1406.5396729, 1337.6610107

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B2_B2_B1

### Relational analysis result of IS_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2042080, upper bound: 843.2041625
time: 0.79 seconds

## Relational analysis of IS_B2_B2_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043052, upper bound: 843.2043052
time: 0.70 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.06 seconds
IS_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2055856, upper bound: 843.2096253
IS_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2056816, upper bound: 843.2091459
IS_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2051141, upper bound: 843.2062918
IS_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2045809, upper bound: 843.2048724
IS_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2041992, upper bound: 843.2045982
IS_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2043420, upper bound: 843.2046954
IS_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2039852, upper bound: 843.2041625
IS_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2043380, upper bound: 843.2044572
IS_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2022760, upper bound: 843.2009625
IS_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2022765, upper bound: 843.2007391
IS_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2047679, upper bound: 843.2053794
IS_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2044335, upper bound: 843.2044427
IS_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2037136, upper bound: 843.2029572
IS_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2038563, upper bound: 843.2030544
IS_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2042080, upper bound: 843.2041625
IS_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.06
Output dim: 0, lower bound: -843.2043052, upper bound: 843.2043052

## BFS IS instance: IS_B2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -166.7506561, 566.8345337, -771.5321045, 863.5733643
1: -334.7795410, 851.1083374, -273.8630371, 692.2714233, -1027.0509033, 1124.9714355
2: -233.0683441, 900.9995117, -189.7585602, 732.1257935, -965.1941528, 1090.7580566
3: -594.6608887, 866.9309082, -484.8710327, 705.5909424, -1300.2517090, 1351.8020020
4: -370.8992920, 924.7813721, -301.9520569, 751.3421631, -1122.2414551, 1226.7333984

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055609, upper bound: 843.2096253
time: 0.74 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054786, upper bound: 843.2088476
time: 0.82 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -204.2186432, 695.2222900, -206.3533173, 701.1197510, -905.3383789, 901.5756226
1: -333.9761353, 849.1367798, -341.9118652, 855.2875977, -1189.2635498, 1191.0485840
2: -232.5206299, 898.9473267, -234.9548645, 904.7566528, -1137.2770996, 1133.9020996
3: -593.2715454, 864.9066772, -603.3047485, 871.6170044, -1464.8884277, 1468.2114258
4: -370.0260010, 922.6676025, -372.8730469, 928.8322754, -1298.8582764, 1295.5404053

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B1_B2_B1

### Relational analysis result of IS_B2_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056037, upper bound: 843.2091459
time: 0.73 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2_B2

### Relational analysis result of IS_B2_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056667, upper bound: 843.2088855
time: 0.74 seconds

## BFS IS instance: IS_B2_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -204.4359436, 695.9414062, -196.5160065, 667.9773560, -872.4132690, 892.4573364
1: -334.3552856, 850.0263062, -322.7358398, 815.6779175, -1150.0327148, 1172.7622070
2: -232.7866821, 899.8817139, -223.8640594, 863.1337891, -1095.9204102, 1123.7457275
3: -593.9490356, 865.8302002, -571.4177856, 831.2780151, -1425.2270508, 1437.2480469
4: -370.4492188, 923.6339111, -356.1497192, 885.9676514, -1256.4168701, 1279.7835693

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2051141, upper bound: 843.2062918
time: 0.75 seconds

## Relational analysis of IS_B2_B1_B1_B2_B1_B2

### Relational analysis result of IS_B2_B1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2050341, upper bound: 843.2061448
time: 0.87 seconds

## BFS IS instance: IS_B2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -203.7614441, 693.6705933, -190.9664917, 649.2957764, -853.0572510, 884.6370239
1: -333.1645508, 847.2582397, -312.2252502, 793.2742310, -1126.4387207, 1159.4835205
2: -232.0157623, 896.9832153, -217.5521393, 839.8068237, -1071.8223877, 1114.5352783
3: -591.9594727, 863.0073853, -554.9062500, 808.4904785, -1400.4497070, 1417.9135742
4: -369.2363281, 920.6639404, -346.5111084, 862.2286377, -1231.4649658, 1267.1750488

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B2_B2_A1

### Relational analysis result of IS_B2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1986754, upper bound: 843.2027773
time: 0.72 seconds

## Relational analysis of IS_B2_B1_B1_B2_B2_A2

### Relational analysis result of IS_B2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2045809, upper bound: 843.2048724
time: 1.21 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -200.2141113, 681.5419312, -208.2344666, 709.2060547, -909.4201660, 889.7763672
1: -327.4259338, 832.5192871, -342.3541565, 866.0283203, -1193.4542236, 1174.8734131
2: -227.9246368, 881.1295166, -236.9849396, 915.6755981, -1143.6002197, 1118.1143799
3: -581.4634399, 847.9473877, -604.9898682, 882.4639893, -1463.9274902, 1452.9372559
4: -362.7260132, 904.3414307, -376.6349182, 939.6113281, -1302.3374023, 1280.9763184

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2014636, upper bound: 843.2028376
time: 0.76 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2041992, upper bound: 843.2045982
time: 0.75 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -198.0664825, 674.6909180, -207.5074310, 706.7034912, -904.7698975, 882.1983032
1: -323.0017700, 824.3739014, -341.0924683, 863.0421143, -1186.0439453, 1165.4661865
2: -225.5462036, 873.0185547, -236.1672668, 912.5221558, -1138.0683594, 1109.1857910
3: -575.2806396, 839.6679077, -602.8424072, 879.4342041, -1454.7148438, 1442.5102539
4: -359.2858276, 896.1059570, -375.3714905, 936.3830566, -1295.6689453, 1271.4774170

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1984364, upper bound: 843.2026004
time: 0.72 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043420, upper bound: 843.2046954
time: 0.89 seconds

## BFS IS instance: IS_B2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -204.1402130, 694.9329224, -214.9245911, 730.2815552, -934.4217529, 909.8574219
1: -333.8447266, 848.7954712, -353.4632874, 891.9364624, -1225.7812500, 1202.2586670
2: -232.4510498, 898.5988159, -244.5974579, 943.1981812, -1175.6491699, 1143.1962891
3: -593.0841675, 864.5813599, -623.9072266, 908.7967529, -1501.8808594, 1488.4885254
4: -369.9197693, 922.3253784, -388.8797302, 967.6702271, -1337.5899658, 1311.2050781

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_B1_A1

### Relational analysis result of IS_B2_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1980797, upper bound: 843.2020674
time: 0.73 seconds

## Relational analysis of IS_B2_B1_B2_B2_B1_A2

### Relational analysis result of IS_B2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2039852, upper bound: 843.2041625
time: 0.77 seconds

## BFS IS instance: IS_B2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -203.4708252, 692.6787109, -211.0598297, 717.4961548, -920.9667358, 903.7385254
1: -332.6619568, 846.0473633, -345.2424011, 876.8836670, -1209.5456543, 1191.2897949
2: -231.6853943, 895.7214355, -240.1630859, 927.5386963, -1159.2241211, 1135.8845215
3: -591.1071777, 861.7792969, -611.7314453, 893.2155762, -1484.3227539, 1473.5107422
4: -368.7153931, 919.3774414, -382.2514343, 951.3140259, -1320.0292969, 1301.6287842

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_B2_A1

### Relational analysis result of IS_B2_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1984324, upper bound: 843.2023621
time: 0.72 seconds

## Relational analysis of IS_B2_B1_B2_B2_B2_A2

### Relational analysis result of IS_B2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043380, upper bound: 843.2044572
time: 1.10 seconds

## BFS IS instance: IS_B2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -204.3803406, 695.7320557, -199.2334137, 676.8768311, -881.2571411, 894.9654541
1: -334.2254639, 849.7663574, -325.4667664, 826.5905762, -1160.8157959, 1175.2331543
2: -232.7091522, 899.6076660, -226.7368774, 874.6265259, -1107.3356934, 1126.3444824
3: -593.7491455, 865.5722046, -577.7370605, 843.0429688, -1436.7921143, 1443.3090820
4: -370.3317871, 923.3668213, -360.6932983, 897.8665161, -1268.1982422, 1284.0600586

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B1_B1_A1

### Relational analysis result of IS_B2_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1963705, upper bound: 843.1988675
time: 0.82 seconds

## Relational analysis of IS_B2_B2_B1_B1_B1_A2

### Relational analysis result of IS_B2_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022760, upper bound: 843.2009625
time: 0.94 seconds

## BFS IS instance: IS_B2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -203.6944427, 693.4503174, -195.2670898, 663.8367310, -867.5310059, 888.7174072
1: -333.0381775, 846.9934082, -318.6680908, 811.1159668, -1144.1540527, 1165.6614990
2: -231.9401398, 896.7010498, -222.4772034, 858.8018799, -1090.7420654, 1119.1778564
3: -591.7644653, 862.7425537, -567.3615112, 827.2867432, -1419.0512695, 1430.1038818
4: -369.1217346, 920.3831177, -354.5248108, 881.9694214, -1251.0910645, 1274.9079590

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B1_B2_A1

### Relational analysis result of IS_B2_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1963709, upper bound: 843.1986441
time: 1.02 seconds

## Relational analysis of IS_B2_B2_B1_B1_B2_A2

### Relational analysis result of IS_B2_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022765, upper bound: 843.2007391
time: 0.77 seconds

## BFS IS instance: IS_B2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -204.1289825, 694.8851318, -217.6750641, 740.6420898, -944.7710571, 912.5601196
1: -333.8211365, 848.7246094, -356.5542297, 904.1804810, -1238.0015869, 1205.2785645
2: -232.4389191, 898.5305786, -247.9568329, 956.6083374, -1189.0472412, 1146.4874268
3: -593.0671387, 864.5166016, -631.9298706, 922.2366943, -1515.3038330, 1496.4465332
4: -369.8994141, 922.2689819, -394.5550232, 981.9003296, -1351.7998047, 1316.8237305

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B2_B1_B1

### Relational analysis result of IS_B2_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047679, upper bound: 843.2053794
time: 0.77 seconds

## Relational analysis of IS_B2_B2_B1_B2_B1_B2

### Relational analysis result of IS_B2_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047677, upper bound: 843.2053445
time: 0.89 seconds

## BFS IS instance: IS_B2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -203.4581909, 692.6297607, -212.5017242, 723.1030273, -926.5612183, 905.1314697
1: -332.6437683, 845.9754639, -347.3486328, 883.1976929, -1215.8414307, 1193.3238525
2: -231.6720428, 895.6511230, -242.2019653, 935.0767212, -1166.7487793, 1137.8530273
3: -591.0894165, 861.7149658, -617.4035645, 900.7084351, -1491.7977295, 1479.1182861
4: -368.6952515, 919.3208618, -385.8279114, 960.1484375, -1328.8437500, 1305.1488037

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B2_B2_B1

### Relational analysis result of IS_B2_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2035626, upper bound: 843.2008090
time: 0.77 seconds

## Relational analysis of IS_B2_B2_B1_B2_B2_B2

### Relational analysis result of IS_B2_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2044335, upper bound: 843.2044427
time: 0.74 seconds

## BFS IS instance: IS_B2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -199.8864441, 680.4214478, -214.8024750, 729.3087769, -929.1951904, 895.2238770
1: -326.8520813, 831.1392822, -351.4609070, 890.5057983, -1217.3579102, 1182.6002197
2: -227.5532990, 879.6986694, -244.3223114, 942.1585693, -1169.7119141, 1124.0209961
3: -580.5203247, 846.5477905, -622.3866577, 908.0667114, -1488.5870361, 1468.9340820
4: -362.1394043, 902.8844604, -388.4810791, 967.4044189, -1329.5438232, 1291.3654785

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B2_B1_A1_A1

### Relational analysis result of IS_B2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2009780, upper bound: 843.2011966
time: 0.75 seconds

## Relational analysis of IS_B2_B2_B2_B1_A1_A2

### Relational analysis result of IS_B2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2037136, upper bound: 843.2029572
time: 0.96 seconds

## BFS IS instance: IS_B2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -197.8595734, 673.9595947, -214.1435089, 727.1048584, -924.9644165, 888.1030884
1: -322.6831970, 823.4678345, -350.3370361, 887.8371582, -1210.5200195, 1173.8046875
2: -225.3132935, 872.0787964, -243.5872345, 939.3717651, -1164.6848145, 1115.6660156
3: -574.7191772, 838.7602539, -620.5079346, 905.3418579, -1480.0610352, 1459.2681885
4: -358.9227905, 895.1704712, -387.3353271, 964.5445557, -1323.4672852, 1282.5058594

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B2_B1_A2_B1

### Relational analysis result of IS_B2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2028026, upper bound: 843.2025379
time: 0.75 seconds

## Relational analysis of IS_B2_B2_B2_B1_A2_B2

### Relational analysis result of IS_B2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2038563, upper bound: 843.2030544
time: 0.72 seconds

## BFS IS instance: IS_B2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -203.8349304, 693.8806152, -225.8131714, 767.3947754, -971.2297363, 919.6937866
1: -333.3126221, 847.4983521, -370.3596191, 936.8298950, -1270.1425781, 1217.8576660
2: -232.1045227, 897.2526245, -257.0593872, 991.2166138, -1223.3211670, 1154.3120117
3: -592.2056274, 863.2735596, -655.1589355, 955.2259521, -1547.4316406, 1518.4322510
4: -369.3724060, 920.9666748, -408.7800293, 1017.5191040, -1386.8914795, 1329.7467041

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B2_B2_B1_B1

### Relational analysis result of IS_B2_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2024474, upper bound: 843.2014268
time: 0.81 seconds

## Relational analysis of IS_B2_B2_B2_B2_B1_B2

### Relational analysis result of IS_B2_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2042080, upper bound: 843.2041625
time: 1.10 seconds

## BFS IS instance: IS_B2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -203.1686707, 691.6444092, -223.5709076, 759.7822876, -962.9507446, 915.2152100
1: -332.1451416, 844.7719116, -365.1613770, 927.9921875, -1260.1370850, 1209.9332275
2: -231.3430634, 894.3964844, -254.5597382, 982.5826416, -1213.9252930, 1148.9561768
3: -590.2411499, 860.4929810, -648.4039917, 946.1370850, -1536.3781738, 1508.8967285
4: -368.1767578, 918.0405884, -405.2694702, 1008.8109741, -1376.9877930, 1323.3100586

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B2_B2_B2_A1

### Relational analysis result of IS_B2_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1983997, upper bound: 843.2022101
time: 0.73 seconds

## Relational analysis of IS_B2_B2_B2_B2_B2_A2

### Relational analysis result of IS_B2_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043052, upper bound: 843.2043052
time: 0.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.18 seconds
IS_B2_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2055609, upper bound: 843.2096253
IS_B2_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2054786, upper bound: 843.2088476
IS_B2_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2056037, upper bound: 843.2091459
IS_B2_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2056667, upper bound: 843.2088855
IS_B2_B1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2051141, upper bound: 843.2062918
IS_B2_B1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2050341, upper bound: 843.2061448
IS_B2_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.1986754, upper bound: 843.2027773
IS_B2_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2045809, upper bound: 843.2048724
IS_B2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2014636, upper bound: 843.2028376
IS_B2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2041992, upper bound: 843.2045982
IS_B2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.1984364, upper bound: 843.2026004
IS_B2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2043420, upper bound: 843.2046954
IS_B2_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.1980797, upper bound: 843.2020674
IS_B2_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2039852, upper bound: 843.2041625
IS_B2_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.1984324, upper bound: 843.2023621
IS_B2_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2043380, upper bound: 843.2044572
IS_B2_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.1963705, upper bound: 843.1988675
IS_B2_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2022760, upper bound: 843.2009625
IS_B2_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.1963709, upper bound: 843.1986441
IS_B2_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2022765, upper bound: 843.2007391
IS_B2_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2047679, upper bound: 843.2053794
IS_B2_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2047677, upper bound: 843.2053445
IS_B2_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2035626, upper bound: 843.2008090
IS_B2_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2044335, upper bound: 843.2044427
IS_B2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2009780, upper bound: 843.2011966
IS_B2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2037136, upper bound: 843.2029572
IS_B2_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2028026, upper bound: 843.2025379
IS_B2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2038563, upper bound: 843.2030544
IS_B2_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2024474, upper bound: 843.2014268
IS_B2_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2042080, upper bound: 843.2041625
IS_B2_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.1983997, upper bound: 843.2022101
IS_B2_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.18
Output dim: 0, lower bound: -843.2043052, upper bound: 843.2043052

## BFS IS instance: IS_B2_B1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -160.9017792, 547.1442261, -751.8417969, 857.7246094
1: -334.7795410, 851.1083374, -264.6123962, 668.0552979, -1002.8348389, 1115.7207031
2: -233.0683441, 900.9995117, -183.0828705, 706.5284424, -939.5968018, 1084.0823975
3: -594.6608887, 866.9309082, -468.1960754, 680.8927612, -1275.5537109, 1335.1269531
4: -370.8992920, 924.7813721, -291.1977539, 724.9754028, -1095.8747559, 1215.9788818

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055433, upper bound: 843.2095953
time: 0.77 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2049450, upper bound: 843.2090098
time: 1.07 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2052863, upper bound: 843.2091626
time: 0.71 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -204.4420166, 695.9404297, -192.4331360, 655.0974121, -859.5393677, 888.3735352
1: -334.3605347, 850.0348511, -316.9958496, 800.3209839, -1134.6815186, 1167.0307617
2: -232.7776031, 899.8623657, -218.9640350, 845.3744507, -1078.1518555, 1118.8264160
3: -593.9290771, 865.8433838, -559.7864990, 815.6773071, -1409.6064453, 1425.6298828
4: -370.4389648, 923.6232300, -347.8560791, 866.6667480, -1237.1053467, 1271.4792480

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A1

### Relational analysis result of IS_B2_B1_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054786, upper bound: 843.2088476
time: 0.72 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A2

### Relational analysis result of IS_B2_B1_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054786, upper bound: 843.2088476
time: 0.85 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -204.2186432, 695.2222900, -199.2061310, 676.8479614, -881.0665894, 894.4283447
1: -333.9761353, 849.1367798, -330.5152588, 825.7000122, -1159.6756592, 1179.6520996
2: -232.5206299, 898.9473267, -226.8559570, 873.2833862, -1105.8039551, 1125.8032227
3: -593.2715454, 864.9066772, -582.7919922, 841.5167847, -1434.7883301, 1447.6984863
4: -370.0260010, 922.6676025, -359.9604492, 896.4665527, -1266.4925537, 1282.6280518

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2053423, upper bound: 843.2088482
time: 0.99 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_B2

### Relational analysis result of IS_B2_B1_B1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047557, upper bound: 843.2067988
time: 0.76 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -203.9819641, 694.4083252, -231.1629791, 787.8473511, -991.8293457, 925.5712891
1: -333.5897522, 848.1460571, -383.1717529, 961.0306396, -1294.6203613, 1231.3177490
2: -232.2513733, 897.8973999, -263.2139282, 1016.5223389, -1248.7736816, 1161.1108398
3: -592.5943604, 863.9020996, -675.9879150, 979.0787354, -1571.6730957, 1539.8898926
4: -369.5994568, 921.5980225, -417.5289612, 1042.7260742, -1412.3255615, 1339.1269531

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_B1

### Relational analysis result of IS_B2_B1_B1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054110, upper bound: 843.2085655
time: 1.04 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_B2

### Relational analysis result of IS_B2_B1_B1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054413, upper bound: 843.2079983
time: 1.12 seconds

## BFS IS instance: IS_B2_B1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -204.4359436, 695.9414062, -191.8447113, 652.1041260, -856.5399780, 887.7860718
1: -334.3552856, 850.0263062, -315.3581543, 796.3602295, -1130.7152100, 1165.3845215
2: -232.7866821, 899.8817139, -218.5736694, 842.5200195, -1075.3065186, 1118.4550781
3: -593.9490356, 865.8302002, -558.1611938, 811.6680908, -1405.6171875, 1423.9914551
4: -370.4492188, 923.6339111, -347.7143860, 864.9199219, -1235.3691406, 1271.3482666

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B1_B2_B1_B1_A1

### Relational analysis result of IS_B2_B1_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047118, upper bound: 843.2057190
time: 0.73 seconds

## Relational analysis of IS_B2_B1_B1_B2_B1_B1_A2

### Relational analysis result of IS_B2_B1_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2051141, upper bound: 843.2062918
time: 0.75 seconds

## BFS IS instance: IS_B2_B1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -203.7535400, 693.6701050, -191.6892853, 650.9740601, -854.7275391, 885.3593140
1: -333.1862793, 847.2308960, -314.1464844, 795.2619019, -1128.4482422, 1161.3773193
2: -232.0097961, 896.9732056, -218.3288879, 841.4931030, -1073.5029297, 1115.3021240
3: -591.9580078, 862.9741821, -556.8192139, 810.5575562, -1402.5156250, 1419.7930908
4: -369.2177124, 920.6436157, -347.4680176, 863.6802368, -1232.8978271, 1268.1113281

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B1_B2_B1_B2_A1

### Relational analysis result of IS_B2_B1_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2046318, upper bound: 843.2055720
time: 0.69 seconds

## Relational analysis of IS_B2_B1_B1_B2_B1_B2_A2

### Relational analysis result of IS_B2_B1_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2050341, upper bound: 843.2061448
time: 0.83 seconds

## BFS IS instance: IS_B2_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -199.1560516, 677.9144287, -190.9664917, 649.2957764, -848.4518433, 868.8808594
1: -325.8258972, 828.0796509, -312.2252502, 793.2742310, -1119.1000977, 1140.3049316
2: -226.7742920, 876.5225220, -217.5521393, 839.8068237, -1066.5810547, 1094.0744629
3: -578.7689819, 843.5424194, -554.9062500, 808.4904785, -1387.2590332, 1398.4487305
4: -360.8907776, 899.7756958, -346.5111084, 862.2286377, -1223.1193848, 1246.2867432

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B1_B2_B2_A1_B1

### Relational analysis result of IS_B2_B1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1979714, upper bound: 843.2022384
time: 1.07 seconds

## Relational analysis of IS_B2_B1_B1_B2_B2_A1_B2

### Relational analysis result of IS_B2_B1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1986754, upper bound: 843.2027773
time: 0.72 seconds

## BFS IS instance: IS_B2_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -201.1445312, 684.2304688, -190.3037415, 647.1022339, -848.2467651, 874.5341187
1: -328.4543457, 836.1271973, -311.1139221, 790.5885010, -1119.0428467, 1147.2410889
2: -229.0332184, 885.0963135, -216.7981720, 836.9843750, -1066.0175781, 1101.8942871
3: -583.9461670, 851.8654785, -552.9840698, 805.7401733, -1389.6862793, 1404.8494873
4: -364.6900635, 908.5225830, -345.3149719, 859.3165283, -1224.0065918, 1253.8375244

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B1_B2_B2_A2_B1

### Relational analysis result of IS_B2_B1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2038769, upper bound: 843.2043335
time: 0.84 seconds

## Relational analysis of IS_B2_B1_B1_B2_B2_A2_B2

### Relational analysis result of IS_B2_B1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2045809, upper bound: 843.2048724
time: 0.79 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -195.6883545, 666.0941772, -208.2344666, 709.2060547, -904.8943481, 874.3286133
1: -320.2035217, 813.6901855, -342.3541565, 866.0283203, -1186.2318115, 1156.0443115
2: -222.7598267, 861.0283203, -236.9849396, 915.6755981, -1138.4354248, 1098.0133057
3: -568.4677734, 828.8347778, -604.9898682, 882.4639893, -1450.9317627, 1433.8247070
4: -354.5043335, 883.8416138, -376.6349182, 939.6113281, -1294.1157227, 1260.4765625

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_B1

### Relational analysis result of IS_B2_B1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2009120, upper bound: 843.2026565
time: 0.81 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_A1_B2

### Relational analysis result of IS_B2_B1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2014636, upper bound: 843.2028376
time: 0.75 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -197.1831970, 670.8043823, -207.5069275, 706.7255249, -903.9086304, 878.3110962
1: -322.0203247, 819.8275757, -341.1342773, 863.0391846, -1185.0594482, 1160.9615479
2: -224.5143890, 867.6148682, -236.1685028, 912.5439453, -1137.0583496, 1103.7830811
3: -572.2965698, 835.2121582, -602.9028320, 879.4425049, -1451.7387695, 1438.1149902
4: -357.5109863, 890.4838867, -375.3633423, 936.4133911, -1293.9243164, 1265.8471680

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_B2_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2040475, upper bound: 843.2042033
time: 0.84 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_B2_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2041992, upper bound: 843.2045982
time: 0.72 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -192.6224213, 656.2145996, -207.5074310, 706.7034912, -899.3258057, 863.7219849
1: -314.1768494, 801.8662720, -341.0924683, 863.0421143, -1177.2189941, 1142.9586182
2: -219.3672485, 849.1054688, -236.1672668, 912.5221558, -1131.8894043, 1085.2727051
3: -559.6428223, 816.7752686, -602.8424072, 879.4342041, -1439.0770264, 1419.6176758
4: -349.4602966, 871.6268921, -375.3714905, 936.3830566, -1285.8433838, 1246.9984131

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B1

### Relational analysis result of IS_B2_B1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1978849, upper bound: 843.2024193
time: 0.72 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_B2

### Relational analysis result of IS_B2_B1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1984364, upper bound: 843.2026004
time: 0.74 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -200.7121124, 682.9157104, -206.7741547, 704.2067871, -904.9187622, 889.6898804
1: -327.4120178, 834.5734253, -339.8629456, 860.0316772, -1187.4434814, 1174.4364014
2: -228.5498047, 883.6110229, -235.3443298, 909.3682861, -1137.9180908, 1118.9553223
3: -582.6856689, 850.3053589, -600.7385864, 876.3909912, -1459.0765381, 1451.0435791
4: -364.0062561, 906.9984131, -374.0898743, 933.1628418, -1297.1690674, 1281.0882568

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B1_A2_A2_A1

### Relational analysis result of IS_B2_B1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2039397, upper bound: 843.2041227
time: 1.06 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2_A2_A2

### Relational analysis result of IS_B2_B1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043420, upper bound: 843.2046954
time: 0.87 seconds

## BFS IS instance: IS_B2_B1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -199.5471802, 679.2030640, -214.9245911, 730.2815552, -929.8287354, 894.1276245
1: -326.5130005, 829.6390991, -353.4632874, 891.9364624, -1218.4493408, 1183.1024170
2: -227.2132416, 878.1655273, -244.5974579, 943.1981812, -1170.4113770, 1122.7629395
3: -579.9037476, 845.1433105, -623.9072266, 908.7967529, -1488.7004395, 1469.0504150
4: -361.5793762, 901.4728394, -388.8797302, 967.6702271, -1329.2496338, 1290.3525391

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B2_B1_A1_B1

### Relational analysis result of IS_B2_B1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1979075, upper bound: 843.2020262
time: 0.97 seconds

## Relational analysis of IS_B2_B1_B2_B2_B1_A1_B2

### Relational analysis result of IS_B2_B1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1980797, upper bound: 843.2020674
time: 1.12 seconds

## BFS IS instance: IS_B2_B1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -201.4362640, 685.2137451, -214.3221283, 728.2276001, -929.6638184, 899.5358276
1: -328.9717102, 837.3174438, -352.4273987, 889.4498291, -1218.4215088, 1189.7448730
2: -229.3682251, 886.3542480, -243.9111481, 940.5864258, -1169.9544678, 1130.2652588
3: -584.8016357, 853.0805054, -622.1343384, 906.2554932, -1491.0571289, 1475.2147217
4: -365.2142944, 909.8139038, -387.7999573, 964.9714966, -1330.1857910, 1297.6138916

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B2_B1_A2_B1

### Relational analysis result of IS_B2_B1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2038131, upper bound: 843.2041212
time: 0.95 seconds

## Relational analysis of IS_B2_B1_B2_B2_B1_A2_B2

### Relational analysis result of IS_B2_B1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2039852, upper bound: 843.2041625
time: 0.80 seconds

## BFS IS instance: IS_B2_B1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -198.8586121, 676.9146729, -211.0598297, 717.4961548, -916.3547363, 887.9744873
1: -325.3196716, 826.8600464, -345.2424011, 876.8836670, -1202.2033691, 1172.1024170
2: -226.4415283, 875.2508545, -240.1630859, 927.5386963, -1153.9802246, 1115.4139404
3: -577.9114380, 842.3047485, -611.7314453, 893.2155762, -1471.1269531, 1454.0361328
4: -360.3659973, 898.4782104, -382.2514343, 951.3140259, -1311.6800537, 1280.7294922

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_B1

### Relational analysis result of IS_B2_B1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1979033, upper bound: 843.2021236
time: 1.00 seconds

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_B2

### Relational analysis result of IS_B2_B1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1984324, upper bound: 843.2023621
time: 0.72 seconds

## BFS IS instance: IS_B2_B1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -200.9110870, 683.4534302, -210.4844208, 715.6015625, -916.5125732, 893.9378052
1: -328.0519714, 835.1826782, -344.2563171, 874.5667725, -1202.6186523, 1179.4387207
2: -228.7675934, 884.1008911, -239.5074615, 925.1036377, -1153.8712158, 1123.6082764
3: -583.2633057, 850.8995972, -610.0482178, 890.8411255, -1474.1043701, 1460.9477539
4: -364.2718506, 907.4979248, -381.2139282, 948.7954102, -1313.0670166, 1288.7119141

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_A1

### Relational analysis result of IS_B2_B1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2039357, upper bound: 843.2038844
time: 0.74 seconds

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_A2

### Relational analysis result of IS_B2_B1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043380, upper bound: 843.2044572
time: 0.72 seconds

## BFS IS instance: IS_B2_B2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -199.7809753, 679.9772339, -199.2334137, 676.8768311, -876.6577148, 879.2105713
1: -326.8746338, 830.5777588, -325.4667664, 826.5905762, -1153.4650879, 1156.0445557
2: -227.4596710, 879.1422729, -226.7368774, 874.6265259, -1102.0861816, 1105.8791504
3: -580.5383911, 846.0957642, -577.7370605, 843.0429688, -1423.5812988, 1423.8322754
4: -361.9740906, 902.4774170, -360.6932983, 897.8665161, -1259.8405762, 1263.1706543

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B2_B1_B1_B1_A1_B1

### Relational analysis result of IS_B2_B2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1959087, upper bound: 843.1973308
time: 0.70 seconds

## Relational analysis of IS_B2_B2_B1_B1_B1_A1_B2

### Relational analysis result of IS_B2_B2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1962327, upper bound: 843.1988629
time: 0.73 seconds

## BFS IS instance: IS_B2_B2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -201.5706329, 685.6584473, -198.2537994, 673.6008301, -875.1714478, 883.9122314
1: -329.2088318, 837.8563843, -323.8626404, 822.6157837, -1151.8244629, 1161.7189941
2: -229.5243988, 886.9261475, -225.6293488, 870.3764648, -1099.9008789, 1112.5552979
3: -585.2083740, 853.6493530, -574.9527588, 838.9745483, -1424.1828613, 1428.6020508
4: -365.4613647, 910.4138184, -358.9776917, 893.5525513, -1259.0139160, 1269.3913574

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B1_B1_A2_A1

### Relational analysis result of IS_B2_B2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2018737, upper bound: 843.2003898
time: 0.68 seconds

## Relational analysis of IS_B2_B2_B1_B1_B1_A2_A2

### Relational analysis result of IS_B2_B2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022760, upper bound: 843.2009625
time: 0.94 seconds

## BFS IS instance: IS_B2_B2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -199.0940857, 677.6903687, -195.2670898, 663.8367310, -862.9307861, 872.9574585
1: -325.6839905, 827.7999268, -318.6680908, 811.1159668, -1136.7999268, 1146.4680176
2: -226.6897736, 876.2302246, -222.4772034, 858.8018799, -1085.4916992, 1098.7072754
3: -578.5513916, 843.2619629, -567.3615112, 827.2867432, -1405.8381348, 1410.6235352
4: -360.7626038, 899.4872437, -354.5248108, 881.9694214, -1242.7320557, 1254.0120850

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_B2_B1_B1_B2_A1_A1

### Relational analysis result of IS_B2_B2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1954107, upper bound: 843.1986425
time: 0.98 seconds

## Relational analysis of IS_B2_B2_B1_B1_B2_A1_A2

### Relational analysis result of IS_B2_B2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1962331, upper bound: 843.1986441
time: 0.72 seconds

## BFS IS instance: IS_B2_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -201.0463562, 683.8931274, -194.7644501, 662.2080688, -863.2543945, 878.6575317
1: -328.2915955, 835.7128906, -317.8523560, 809.1355591, -1137.4271240, 1153.5649414
2: -228.9244232, 884.6665649, -221.9237213, 856.7321777, -1085.6566162, 1106.5903320
3: -583.6740723, 851.4588623, -565.9745483, 825.2631836, -1408.9372559, 1417.4333496
4: -364.5182495, 908.0966797, -353.6700745, 879.8572388, -1244.3752441, 1261.7666016

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B1_B2_A2_A1

### Relational analysis result of IS_B2_B2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2018742, upper bound: 843.2001664
time: 0.70 seconds

## Relational analysis of IS_B2_B2_B1_B1_B2_A2_A2

### Relational analysis result of IS_B2_B2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022765, upper bound: 843.2007391
time: 0.70 seconds

## BFS IS instance: IS_B2_B2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -204.1289825, 694.8851318, -213.5215912, 726.3967896, -930.5257568, 908.4066772
1: -333.8211365, 848.7246094, -349.8966675, 886.7261963, -1220.5472412, 1198.6209717
2: -232.4389191, 898.5305786, -243.2013397, 938.0737915, -1170.5124512, 1141.7319336
3: -593.0671387, 864.5166016, -619.9860840, 904.5924683, -1497.6596680, 1484.5026855
4: -369.8994141, 922.2689819, -386.9009399, 963.0005493, -1332.8999023, 1309.1699219

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B2_B1_B1_B1

### Relational analysis result of IS_B2_B2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047404, upper bound: 843.2051572
time: 0.78 seconds

## Relational analysis of IS_B2_B2_B1_B2_B1_B1_B2

### Relational analysis result of IS_B2_B2_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047679, upper bound: 843.2053794
time: 0.69 seconds

## BFS IS instance: IS_B2_B2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -203.4483795, 692.6181030, -208.2393951, 708.4486084, -911.8968506, 900.8574829
1: -332.6544189, 845.9343262, -341.0079041, 865.2076416, -1197.8620605, 1186.9422607
2: -231.6635132, 895.6274414, -237.3934937, 915.5729370, -1147.2364502, 1133.0207520
3: -591.0798340, 861.6671143, -605.0063477, 882.5072632, -1473.5869141, 1466.6734619
4: -368.6708069, 919.2853394, -377.8889771, 939.9581909, -1308.6289062, 1297.1743164

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B2_B1_B2_B1

### Relational analysis result of IS_B2_B2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047028, upper bound: 843.2051585
time: 0.77 seconds

## Relational analysis of IS_B2_B2_B1_B2_B1_B2_B2

### Relational analysis result of IS_B2_B2_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047677, upper bound: 843.2053445
time: 0.78 seconds

## BFS IS instance: IS_B2_B2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -203.4581909, 692.6297607, -207.4937134, 706.1360474, -909.5942383, 900.1234741
1: -332.6437683, 845.9754639, -339.3368225, 862.5180054, -1195.1617432, 1185.3121338
2: -231.6720428, 895.6511230, -236.5336609, 913.0335083, -1144.7054443, 1132.1846924
3: -591.0894165, 861.7149658, -603.0602417, 879.7378540, -1470.8271484, 1464.7750244
4: -368.6952515, 919.3208618, -376.8010864, 937.5715942, -1306.2668457, 1296.1219482

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B2_B2_B1_A1

### Relational analysis result of IS_B2_B2_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2031604, upper bound: 843.2002362
time: 0.70 seconds

## Relational analysis of IS_B2_B2_B1_B2_B2_B1_A2

### Relational analysis result of IS_B2_B2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2035626, upper bound: 843.2008090
time: 0.79 seconds

## BFS IS instance: IS_B2_B2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -202.7897491, 690.4083862, -207.5680084, 705.8643188, -908.6540527, 897.9763184
1: -331.5000000, 843.2429199, -338.9636536, 862.4502563, -1193.9501953, 1182.2064209
2: -230.9114532, 892.8042603, -236.5424805, 913.1660767, -1144.0775146, 1129.3466797
3: -589.1353149, 858.9220581, -603.0148315, 879.5731201, -1468.7082520, 1461.9367676
4: -367.4910889, 916.3922729, -376.8496094, 937.7730103, -1305.2640381, 1293.2419434

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B2_B2_B2_A1

### Relational analysis result of IS_B2_B2_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2040312, upper bound: 843.2038699
time: 0.76 seconds

## Relational analysis of IS_B2_B2_B1_B2_B2_B2_A2

### Relational analysis result of IS_B2_B2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2044335, upper bound: 843.2044427
time: 1.28 seconds

## BFS IS instance: IS_B2_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -195.3185883, 664.8258057, -214.8024750, 729.3087769, -924.6273804, 879.6281738
1: -319.5536194, 812.1310425, -351.4609070, 890.5057983, -1210.0594482, 1163.5919189
2: -222.3381348, 859.4121094, -244.3223114, 942.1585693, -1164.4967041, 1103.7343750
3: -567.3969116, 827.2449951, -622.3866577, 908.0667114, -1475.4636230, 1449.6314697
4: -353.8362122, 882.1848755, -388.4810791, 967.4044189, -1321.2406006, 1270.6660156

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B2_B1_A1_A1_B1

### Relational analysis result of IS_B2_B2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1999242, upper bound: 843.2006801
time: 0.73 seconds

## Relational analysis of IS_B2_B2_B2_B1_A1_A1_B2

### Relational analysis result of IS_B2_B2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2009780, upper bound: 843.2011966
time: 0.74 seconds

## BFS IS instance: IS_B2_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -196.7843781, 669.4277344, -214.1313019, 727.1006470, -923.8850098, 883.5590210
1: -321.3546753, 818.1179199, -350.3565979, 887.8113403, -1209.1660156, 1168.4744873
2: -224.0614624, 865.8547974, -243.5754395, 939.3526001, -1163.4140625, 1109.4301758
3: -571.1663818, 833.4816284, -620.5006714, 905.3129883, -1476.4790039, 1453.9822998
4: -356.7825928, 888.7052612, -387.3123779, 964.5125122, -1321.2951660, 1276.0175781

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B2_B1_A1_A2_B1

### Relational analysis result of IS_B2_B2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2026599, upper bound: 843.2024407
time: 0.71 seconds

## Relational analysis of IS_B2_B2_B2_B1_A1_A2_B2

### Relational analysis result of IS_B2_B2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2037136, upper bound: 843.2029572
time: 0.68 seconds

## BFS IS instance: IS_B2_B2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -197.8595734, 673.9595947, -208.4125519, 707.5438843, -905.4034424, 882.3720093
1: -322.6831970, 823.4678345, -341.1378479, 863.9586182, -1186.6418457, 1164.6057129
2: -225.3132935, 872.0787964, -237.0485992, 913.9938354, -1139.3071289, 1109.1274414
3: -574.7191772, 838.7602539, -604.1007690, 881.0826416, -1455.8016357, 1442.8610840
4: -358.9227905, 895.1704712, -376.9185181, 938.6207275, -1297.5434570, 1272.0889893

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B2_B1_A2_B1_A1

### Relational analysis result of IS_B2_B2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1968971, upper bound: 843.2004428
time: 0.73 seconds

## Relational analysis of IS_B2_B2_B2_B1_A2_B1_A2

### Relational analysis result of IS_B2_B2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2028026, upper bound: 843.2025379
time: 0.86 seconds

## BFS IS instance: IS_B2_B2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -197.6553345, 673.2565308, -214.8544464, 729.7517700, -927.4071045, 888.1109009
1: -322.3549194, 822.6129150, -351.3861389, 891.2824097, -1213.6370850, 1173.9990234
2: -225.0821075, 871.1702881, -244.3484039, 942.5444336, -1167.6264648, 1115.5186768
3: -574.1322632, 837.8909912, -622.0427856, 908.5747070, -1482.7070312, 1459.9338379
4: -358.5567017, 894.2391357, -388.6008301, 967.3416138, -1325.8983154, 1282.8399658

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B2_B1_A2_B2_A1

### Relational analysis result of IS_B2_B2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1979508, upper bound: 843.2009594
time: 0.72 seconds

## Relational analysis of IS_B2_B2_B2_B1_A2_B2_A2

### Relational analysis result of IS_B2_B2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2038563, upper bound: 843.2030544
time: 0.81 seconds

## BFS IS instance: IS_B2_B2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -203.8349304, 693.8806152, -221.0416718, 751.1032104, -954.9381104, 914.9221802
1: -333.3126221, 847.4983521, -362.7113953, 916.8981934, -1250.2105713, 1210.2095947
2: -232.1045227, 897.2526245, -251.5912628, 969.9419556, -1202.0465088, 1148.8438721
3: -592.2056274, 863.2735596, -641.3731079, 935.0051880, -1527.2108154, 1504.6464844
4: -369.3724060, 920.9666748, -399.9920349, 995.7994995, -1365.1718750, 1320.9586182

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B2_B2_B1_B1_A1

### Relational analysis result of IS_B2_B2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020451, upper bound: 843.2008541
time: 0.73 seconds

## Relational analysis of IS_B2_B2_B2_B2_B1_B1_A2

### Relational analysis result of IS_B2_B2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2024474, upper bound: 843.2014268
time: 0.90 seconds

## BFS IS instance: IS_B2_B2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -203.1548309, 691.6158447, -221.1070251, 751.9729004, -955.1276855, 912.7228394
1: -332.1475220, 844.7122192, -362.4070129, 918.2570801, -1250.4045410, 1207.1191406
2: -231.3297577, 894.3518677, -251.8039856, 971.5339355, -1202.8636475, 1146.1557617
3: -590.2194214, 860.4282227, -641.3388672, 936.3071289, -1526.5262451, 1501.7666016
4: -368.1452942, 917.9851685, -400.5539551, 996.8090210, -1364.9543457, 1318.5390625

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B2_B2_B1_B2_A1

### Relational analysis result of IS_B2_B2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2038057, upper bound: 843.2035897
time: 0.88 seconds

## Relational analysis of IS_B2_B2_B2_B2_B1_B2_A2

### Relational analysis result of IS_B2_B2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2042080, upper bound: 843.2041625
time: 0.82 seconds

## BFS IS instance: IS_B2_B2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -198.5474548, 675.8421631, -223.5441895, 759.6898193, -958.2372437, 899.3862305
1: -324.7811279, 825.5420532, -365.1163940, 927.8813477, -1252.6624756, 1190.6584473
2: -226.0895538, 873.8820801, -254.5291901, 982.4642334, -1208.5538330, 1128.4111328
3: -577.0189209, 840.9718628, -648.3259888, 946.0242310, -1523.0432129, 1489.2978516
4: -359.8099976, 897.0902710, -405.2214050, 1008.6896362, -1368.4996338, 1302.3115234

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B2_B2_B2_A1_B1

### Relational analysis result of IS_B2_B2_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1978269, upper bound: 843.2018079
time: 0.78 seconds

## Relational analysis of IS_B2_B2_B2_B2_B2_A1_B2

### Relational analysis result of IS_B2_B2_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1983997, upper bound: 843.2022101
time: 0.87 seconds

## BFS IS instance: IS_B2_B2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -200.5518494, 682.2175903, -223.2395325, 758.6918335, -959.2436523, 905.4571533
1: -327.4528809, 833.6545410, -364.6068420, 926.6494141, -1254.1021729, 1198.2613525
2: -228.3601227, 882.5168457, -254.1817627, 981.1848755, -1209.5446777, 1136.6986084
3: -582.2476196, 849.3565063, -647.4520264, 944.7556763, -1527.0031738, 1496.8082275
4: -363.6200867, 905.8966675, -404.6676636, 1007.3725586, -1370.9924316, 1310.5643311

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B2_B2_B2_A2_B1

### Relational analysis result of IS_B2_B2_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2037324, upper bound: 843.2039029
time: 0.76 seconds

## Relational analysis of IS_B2_B2_B2_B2_B2_A2_B2

### Relational analysis result of IS_B2_B2_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043052, upper bound: 843.2043052
time: 0.73 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.56 seconds
IS_B2_B1_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2049450, upper bound: 843.2090098
IS_B2_B1_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2052863, upper bound: 843.2091626
IS_B2_B1_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2054786, upper bound: 843.2088476
IS_B2_B1_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2054786, upper bound: 843.2088476
IS_B2_B1_B1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2053423, upper bound: 843.2088482
IS_B2_B1_B1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2047557, upper bound: 843.2067988
IS_B2_B1_B1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2054110, upper bound: 843.2085655
IS_B2_B1_B1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2054413, upper bound: 843.2079983
IS_B2_B1_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2047118, upper bound: 843.2057190
IS_B2_B1_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2051141, upper bound: 843.2062918
IS_B2_B1_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2046318, upper bound: 843.2055720
IS_B2_B1_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2050341, upper bound: 843.2061448
IS_B2_B1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1979714, upper bound: 843.2022384
IS_B2_B1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1986754, upper bound: 843.2027773
IS_B2_B1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2038769, upper bound: 843.2043335
IS_B2_B1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2045809, upper bound: 843.2048724
IS_B2_B1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2009120, upper bound: 843.2026565
IS_B2_B1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2014636, upper bound: 843.2028376
IS_B2_B1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2040475, upper bound: 843.2042033
IS_B2_B1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2041992, upper bound: 843.2045982
IS_B2_B1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1978849, upper bound: 843.2024193
IS_B2_B1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1984364, upper bound: 843.2026004
IS_B2_B1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2039397, upper bound: 843.2041227
IS_B2_B1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2043420, upper bound: 843.2046954
IS_B2_B1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1979075, upper bound: 843.2020262
IS_B2_B1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1980797, upper bound: 843.2020674
IS_B2_B1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2038131, upper bound: 843.2041212
IS_B2_B1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2039852, upper bound: 843.2041625
IS_B2_B1_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1979033, upper bound: 843.2021236
IS_B2_B1_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1984324, upper bound: 843.2023621
IS_B2_B1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2039357, upper bound: 843.2038844
IS_B2_B1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2043380, upper bound: 843.2044572
IS_B2_B2_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1959087, upper bound: 843.1973308
IS_B2_B2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1962327, upper bound: 843.1988629
IS_B2_B2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2018737, upper bound: 843.2003898
IS_B2_B2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2022760, upper bound: 843.2009625
IS_B2_B2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1954107, upper bound: 843.1986425
IS_B2_B2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1962331, upper bound: 843.1986441
IS_B2_B2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2018742, upper bound: 843.2001664
IS_B2_B2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2022765, upper bound: 843.2007391
IS_B2_B2_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2047404, upper bound: 843.2051572
IS_B2_B2_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2047679, upper bound: 843.2053794
IS_B2_B2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2047028, upper bound: 843.2051585
IS_B2_B2_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2047677, upper bound: 843.2053445
IS_B2_B2_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2031604, upper bound: 843.2002362
IS_B2_B2_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2035626, upper bound: 843.2008090
IS_B2_B2_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2040312, upper bound: 843.2038699
IS_B2_B2_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2044335, upper bound: 843.2044427
IS_B2_B2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1999242, upper bound: 843.2006801
IS_B2_B2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2009780, upper bound: 843.2011966
IS_B2_B2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2026599, upper bound: 843.2024407
IS_B2_B2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2037136, upper bound: 843.2029572
IS_B2_B2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1968971, upper bound: 843.2004428
IS_B2_B2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2028026, upper bound: 843.2025379
IS_B2_B2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1979508, upper bound: 843.2009594
IS_B2_B2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2038563, upper bound: 843.2030544
IS_B2_B2_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2020451, upper bound: 843.2008541
IS_B2_B2_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2024474, upper bound: 843.2014268
IS_B2_B2_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2038057, upper bound: 843.2035897
IS_B2_B2_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2042080, upper bound: 843.2041625
IS_B2_B2_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1978269, upper bound: 843.2018079
IS_B2_B2_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.1983997, upper bound: 843.2022101
IS_B2_B2_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2037324, upper bound: 843.2039029
IS_B2_B2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.56
Output dim: 0, lower bound: -843.2043052, upper bound: 843.2043052

## BFS IS instance: IS_B2_B1_B1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -157.7534027, 536.6477661, -741.3453369, 854.5761719
1: -334.7795410, 851.1083374, -259.4713440, 655.1001587, -989.8796997, 1110.5797119
2: -233.0683441, 900.9995117, -179.4129944, 692.9012451, -925.9696045, 1080.4124756
3: -594.6608887, 866.9309082, -459.0368347, 667.5084229, -1262.1693115, 1325.9676514
4: -370.8992920, 924.7813721, -285.3155518, 710.8181152, -1081.7174072, 1210.0969238

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2049319, upper bound: 843.2089007
time: 0.70 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1_A1

### Relational analysis result of IS_B2_B1_B1_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2015091, upper bound: 843.2052347
time: 0.74 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1_A1

### Relational analysis result of IS_B2_B1_B1_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2006580, upper bound: 843.2071815
time: 0.89 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B1_A2

### Relational analysis result of IS_B2_B1_B1_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2049298, upper bound: 843.2089855
time: 0.73 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -156.1416168, 530.2408447, -734.9384766, 852.9644165
1: -334.7795410, 851.1083374, -256.1047058, 647.9691162, -982.7485962, 1107.2130127
2: -233.0683441, 900.9995117, -177.8142090, 685.1729126, -918.2412720, 1078.8137207
3: -594.6608887, 866.9309082, -454.1234131, 660.6108398, -1255.2717285, 1321.0541992
4: -370.8992920, 924.7813721, -282.9875183, 703.5670776, -1074.4663086, 1207.7689209

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B2_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2052640, upper bound: 843.2090479
time: 0.71 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B2_A1

### Relational analysis result of IS_B2_B1_B1_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020697, upper bound: 843.2054822
time: 0.81 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B2_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2049940, upper bound: 843.2086888
time: 0.77 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_B1_B2_B2

### Relational analysis result of IS_B2_B1_B1_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2051192, upper bound: 843.2084872
time: 0.86 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -195.7605438, 665.7628174, -192.4331360, 655.0974121, -850.8579102, 858.1959229
1: -320.3244629, 813.3563232, -316.9958496, 800.3209839, -1120.6455078, 1130.3521729
2: -222.8683167, 860.7802124, -218.9640350, 845.3744507, -1068.2425537, 1079.7442627
3: -568.6416016, 828.6393433, -559.7864990, 815.6773071, -1384.3188477, 1388.4257812
4: -354.7007141, 883.5136719, -347.8560791, 866.6667480, -1221.3671875, 1231.3695068

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2051950, upper bound: 843.2084359
time: 1.14 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054761, upper bound: 843.2086551
time: 0.87 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=901.5203857421875
rel_dist={0: [-843.2117555388025, 843.2117555388029]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -843.1908690, upper bound: 843.1932989
time: 0.78 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2117348, upper bound: 843.2117348
time: 0.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.68 seconds
IS_B1, status: Status.VERIFIED, split count: 1, time: 1.68
Output dim: 0, lower bound: -843.1908690, upper bound: 843.1932989
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -843.2117348, upper bound: 843.2117348

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -204.6462860, 696.6473389, -901.3449707, 901.4689941
1: -334.7795410, 851.1083374, -334.6973877, 850.8936157, -1185.6730957, 1185.8056641
2: -233.0683441, 900.9995117, -233.0101318, 900.7739868, -1133.8422852, 1134.0096436
3: -594.6608887, 866.9309082, -594.5149536, 866.7125244, -1461.3732910, 1461.4458008
4: -370.8992920, 924.7813721, -370.8064880, 924.5512695, -1295.4505615, 1295.5878906

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2069197, upper bound: 843.2102337
time: 0.70 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2058473, upper bound: 843.2058473
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.45 seconds
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -843.2069197, upper bound: 843.2102337
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -843.2058473, upper bound: 843.2058473

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -195.9719086, 666.4939575, -871.1915283, 892.7947388
1: -334.7795410, 851.1083374, -320.6729126, 814.2442627, -1149.0235596, 1171.7812500
2: -233.0683441, 900.9995117, -223.1084137, 861.7239990, -1094.7923584, 1124.1079102
3: -594.6608887, 866.9309082, -569.2477417, 829.5389404, -1424.1998291, 1436.1787109
4: -370.8992920, 924.7813721, -355.0805054, 884.4744263, -1255.3736572, 1279.8618164

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2067558, upper bound: 843.2101222
time: 0.71 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2061901, upper bound: 843.2067988
time: 0.74 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -203.8545532, 693.9155273, -216.5949707, 736.6307983, -940.4852905, 910.5104980
1: -333.3247070, 847.5336304, -353.7225952, 899.5317993, -1232.8564453, 1201.2562256
2: -232.1188965, 897.2911987, -246.5761871, 952.2543335, -1184.3732910, 1143.8674316
3: -592.2492065, 863.3193970, -628.5303955, 917.1265869, -1509.3756104, 1491.8498535
4: -369.3995667, 921.0283813, -392.4837646, 977.6730347, -1347.0726318, 1313.5120850

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1

### Relational analysis result of IS_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2046742, upper bound: 843.2034344
time: 0.75 seconds

## Relational analysis of IS_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2058473, upper bound: 843.2058473
time: 0.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.63 seconds
IS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -843.2067558, upper bound: 843.2101222
IS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -843.2061901, upper bound: 843.2067988
IS_B2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -843.2046742, upper bound: 843.2034344
IS_B2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -843.2058473, upper bound: 843.2058473

## BFS IS instance: IS_B2_B1_B1

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -188.1023407, 639.3076782, -844.0052490, 884.9249878
1: -334.7795410, 851.1083374, -308.0525208, 781.0581665, -1115.8376465, 1159.1608887
2: -233.0683441, 900.9995117, -214.1279755, 826.3632202, -1059.4315186, 1115.1274414
3: -594.6608887, 866.9309082, -546.4844971, 795.9373779, -1390.5982666, 1413.4152832
4: -370.8992920, 924.7813721, -340.7922058, 848.3249512, -1219.2242432, 1265.5732422

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056805, upper bound: 843.2096253
time: 0.93 seconds

## Relational analysis of IS_B2_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043420, upper bound: 843.2046954
time: 0.65 seconds

## BFS IS instance: IS_B2_B1_B2

### Backsubstitution after applying IS history:
0: -204.0078278, 694.4490356, -206.0993347, 700.5337524, -904.5415649, 900.5483398
1: -333.6326904, 848.1758423, -338.1727295, 855.3402710, -1188.9729004, 1186.3486328
2: -232.3007660, 897.9718628, -234.7609100, 905.4667358, -1137.7674561, 1132.7326660
3: -592.7230835, 863.9620361, -599.1926880, 871.6817627, -1464.4047852, 1463.1544189
4: -369.6755676, 921.6985474, -373.4751282, 929.4934692, -1299.1689453, 1295.1737061

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B2_B1

### Relational analysis result of IS_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2050749, upper bound: 843.2063106
time: 0.72 seconds

## Relational analysis of IS_B2_B1_B2_B2

### Relational analysis result of IS_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043380, upper bound: 843.2044572
time: 0.72 seconds

## BFS IS instance: IS_B2_B2_B1

### Backsubstitution after applying IS history:
0: -203.8545532, 693.9155273, -208.3971863, 708.1631470, -912.0177002, 902.3126831
1: -333.3247070, 847.5336304, -340.4141846, 864.8139648, -1198.1386719, 1187.9477539
2: -232.1188965, 897.2911987, -237.1992188, 915.2454224, -1147.3642578, 1134.4904785
3: -592.2492065, 863.3193970, -604.6151733, 881.9611206, -1474.2102051, 1467.9345703
4: -369.3995667, 921.0283813, -377.4720764, 939.7947388, -1309.1942139, 1298.5002441

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_B2_B1_B1

### Relational analysis result of IS_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022765, upper bound: 843.2009625
time: 0.87 seconds

## Relational analysis of IS_B2_B2_B1_B2

### Relational analysis result of IS_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2038563, upper bound: 843.2030544
time: 1.02 seconds

## BFS IS instance: IS_B2_B2_B2

### Backsubstitution after applying IS history:
0: -203.2580872, 691.8847656, -226.4450531, 770.6193848, -973.8774414, 918.3297119
1: -332.3664551, 845.0065308, -370.8050842, 940.7882080, -1273.1546631, 1215.8114014
2: -231.4593506, 894.6897583, -258.0072632, 995.7050781, -1227.1644287, 1152.6970215
3: -590.5859985, 860.7678833, -657.5964355, 959.4372559, -1550.0231934, 1518.3642578
4: -368.3508301, 918.3982544, -410.5917664, 1022.1001587, -1390.4509277, 1328.9899902

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_B2_B2_B1

### Relational analysis result of IS_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047706, upper bound: 843.2053794
time: 0.81 seconds

## Relational analysis of IS_B2_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043052, upper bound: 843.2043052
time: 0.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.68 seconds
IS_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -843.2056805, upper bound: 843.2096253
IS_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -843.2043420, upper bound: 843.2046954
IS_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -843.2050749, upper bound: 843.2063106
IS_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -843.2043380, upper bound: 843.2044572
IS_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -843.2022765, upper bound: 843.2009625
IS_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -843.2038563, upper bound: 843.2030544
IS_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -843.2047706, upper bound: 843.2053794
IS_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -843.2043052, upper bound: 843.2043052

## BFS IS instance: IS_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -182.7452393, 621.1705933, -825.8681641, 879.5679321
1: -334.7795410, 851.1083374, -299.4613342, 758.8191528, -1093.5983887, 1150.5697021
2: -233.0683441, 900.9995117, -208.0237579, 802.6618042, -1035.7301025, 1109.0233154
3: -594.6608887, 866.9309082, -530.9902954, 773.3157349, -1367.9765625, 1397.9208984
4: -370.8992920, 924.7813721, -331.0804749, 824.0029297, -1194.9022217, 1255.8618164

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055856, upper bound: 843.2096253
time: 0.88 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056805, upper bound: 843.2091459
time: 0.73 seconds

## BFS IS instance: IS_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -203.8449402, 693.9571533, -208.2344666, 709.2060547, -913.0510254, 902.1916504
1: -333.3291626, 847.6120605, -342.3541565, 866.0283203, -1199.3574219, 1189.9660645
2: -232.1156616, 897.3569336, -236.9849396, 915.6755981, -1147.7912598, 1134.3419189
3: -592.2053833, 863.3837280, -604.9898682, 882.4639893, -1474.6694336, 1468.3735352
4: -369.3968811, 921.0657349, -376.6349182, 939.6113281, -1309.0081787, 1297.7006836

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B1_B2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2030434, upper bound: 843.2038196
time: 0.70 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2

### Relational analysis result of IS_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2040469, upper bound: 843.2043548
time: 0.87 seconds

## BFS IS instance: IS_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -204.0078278, 694.4490356, -200.6152496, 681.8596191, -885.8674316, 895.0640259
1: -333.6326904, 848.1758423, -329.4119568, 832.5629883, -1166.1955566, 1177.5875244
2: -232.3007660, 897.9718628, -228.5471344, 881.2005615, -1113.5012207, 1126.5187988
3: -592.7230835, 863.9620361, -583.4202881, 848.5297241, -1441.2525635, 1447.3823242
4: -369.6755676, 921.6985474, -363.5925598, 904.5921021, -1274.2675781, 1285.2911377

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B2_B1_B1

### Relational analysis result of IS_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2050724, upper bound: 843.2062918
time: 0.74 seconds

## Relational analysis of IS_B2_B1_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2045809, upper bound: 843.2048724
time: 0.78 seconds

## BFS IS instance: IS_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -203.2297363, 691.7881470, -218.8594666, 743.5152588, -946.7449341, 910.6475830
1: -332.2830505, 844.9442749, -359.8024902, 908.1167603, -1240.3997803, 1204.7467041
2: -231.4154968, 894.5876465, -249.0902405, 960.5374146, -1191.9528809, 1143.6778564
3: -590.4360962, 860.6875000, -635.2955933, 925.3173828, -1515.7534180, 1495.9831543
4: -368.2861633, 918.2517700, -396.0701904, 985.4878540, -1353.7740479, 1314.3220215

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B2_B2_B1

### Relational analysis result of IS_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2039852, upper bound: 843.2041625
time: 0.80 seconds

## Relational analysis of IS_B2_B1_B2_B2_B2

### Relational analysis result of IS_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043380, upper bound: 843.2044572
time: 0.84 seconds

## BFS IS instance: IS_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -203.8545532, 693.9155273, -203.1571045, 690.2086182, -894.0631104, 897.0726318
1: -333.3247070, 847.5336304, -331.9653015, 842.8645630, -1176.1892090, 1179.4989014
2: -232.1188965, 897.2911987, -231.2273712, 891.8779297, -1123.9968262, 1128.5185547
3: -592.2492065, 863.3193970, -589.3912964, 859.6685181, -1451.9177246, 1452.7106934
4: -369.3995667, 921.0283813, -367.9232788, 915.8019409, -1285.2014160, 1288.9514160

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B2_B1_B1_B1

### Relational analysis result of IS_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2017624, upper bound: 843.1993435
time: 1.02 seconds

## Relational analysis of IS_B2_B2_B1_B1_B2

### Relational analysis result of IS_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020868, upper bound: 843.2008756
time: 0.97 seconds

## BFS IS instance: IS_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -203.0132294, 691.0938110, -214.5763245, 728.5659180, -931.5791016, 905.6699829
1: -331.8992004, 844.0873413, -351.0879517, 889.6018677, -1221.5010986, 1195.1751709
2: -231.1756592, 893.6990356, -244.0693817, 941.2169800, -1172.3925781, 1137.7679443
3: -589.8231201, 859.8167725, -621.7396240, 907.1300049, -1496.9531250, 1481.5563965
4: -367.9129639, 917.3610840, -388.0875549, 966.4251709, -1334.3381348, 1305.4486084

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_B2_B1_B2_A1

### Relational analysis result of IS_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2025390, upper bound: 843.2022453
time: 0.72 seconds

## Relational analysis of IS_B2_B2_B1_B2_A2

### Relational analysis result of IS_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2035426, upper bound: 843.2027805
time: 0.78 seconds

## BFS IS instance: IS_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -203.2580872, 691.8847656, -221.1631470, 752.5880127, -955.8460083, 913.0479126
1: -332.3664551, 845.0065308, -362.3185730, 918.7485352, -1251.1149902, 1207.3250732
2: -231.4593506, 894.6897583, -251.9917755, 972.2334595, -1203.6928711, 1146.6815186
3: -590.5859985, 860.7678833, -642.3211060, 937.0757446, -1527.6617432, 1503.0889893
4: -368.3508301, 918.3982544, -401.0153809, 998.0183716, -1366.3691406, 1319.4135742

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B2_B1_B1

### Relational analysis result of IS_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2047679, upper bound: 843.2053794
time: 0.91 seconds

## Relational analysis of IS_B2_B2_B2_B1_B2

### Relational analysis result of IS_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2044335, upper bound: 843.2044427
time: 0.80 seconds

## BFS IS instance: IS_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -202.4833527, 689.2477417, -229.3682861, 779.4718628, -981.9552002, 918.6160278
1: -331.0305786, 841.8048706, -376.1909790, 951.6137695, -1282.6442871, 1217.9957275
2: -230.5783539, 891.3328857, -261.1899414, 1007.1529541, -1237.7313232, 1152.5225830
3: -588.3127441, 857.5153198, -665.7946777, 970.3303833, -1558.6430664, 1523.3096924
4: -366.9697876, 914.9727173, -415.4416199, 1033.9396973, -1400.9091797, 1330.4143066

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B2_B2_A1

### Relational analysis result of IS_B2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -843.1728016, upper bound: 843.1838577
time: 0.99 seconds

## Relational analysis of IS_B2_B2_B2_B2_A2

### Relational analysis result of IS_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1728016, upper bound: 843.2043052
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.67 seconds
IS_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2055856, upper bound: 843.2096253
IS_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2056805, upper bound: 843.2091459
IS_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2030434, upper bound: 843.2038196
IS_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2040469, upper bound: 843.2043548
IS_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2050724, upper bound: 843.2062918
IS_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2045809, upper bound: 843.2048724
IS_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2039852, upper bound: 843.2041625
IS_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2043380, upper bound: 843.2044572
IS_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2017624, upper bound: 843.1993435
IS_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2020868, upper bound: 843.2008756
IS_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2025390, upper bound: 843.2022453
IS_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2035426, upper bound: 843.2027805
IS_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2047679, upper bound: 843.2053794
IS_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.2044335, upper bound: 843.2044427
IS_B2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.1728016, upper bound: 843.1838577
IS_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 0, lower bound: -843.1728016, upper bound: 843.2043052

## BFS IS instance: IS_B2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -166.7506561, 566.8345337, -771.5321045, 863.5733643
1: -334.7795410, 851.1083374, -273.8630371, 692.2714233, -1027.0509033, 1124.9714355
2: -233.0683441, 900.9995117, -189.7585602, 732.1257935, -965.1941528, 1090.7580566
3: -594.6608887, 866.9309082, -484.8710327, 705.5909424, -1300.2517090, 1351.8020020
4: -370.8992920, 924.7813721, -301.9520569, 751.3421631, -1122.2414551, 1226.7333984

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_B1_B1_B1_B1_A1

### Relational analysis result of IS_B2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055856, upper bound: 843.2096253
time: 1.13 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_A2

### Relational analysis result of IS_B2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055856, upper bound: 843.2096253
time: 0.92 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -203.3523560, 692.3411865, -206.3533173, 701.1197510, -904.4720459, 898.6945190
1: -332.5233765, 845.5808716, -341.9118652, 855.2875977, -1187.8109131, 1187.4926758
2: -231.5299683, 895.2566528, -234.9548645, 904.7566528, -1136.2863770, 1130.2113037
3: -590.7614136, 861.2525635, -603.3047485, 871.6170044, -1462.3784180, 1464.5573730
4: -368.4442444, 918.8638916, -372.8730469, 928.8322754, -1297.2764893, 1291.7365723

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B1_B2_B1

### Relational analysis result of IS_B2_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056037, upper bound: 843.2091459
time: 0.76 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2_B2

### Relational analysis result of IS_B2_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056666, upper bound: 843.2088855
time: 0.93 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -199.9364319, 680.6033325, -208.2344666, 709.2060547, -909.1424561, 888.8377686
1: -326.9184875, 831.3753052, -342.3541565, 866.0283203, -1192.9466553, 1173.7293701
2: -227.5901337, 880.0817871, -236.9849396, 915.6755981, -1143.2657471, 1117.0667725
3: -580.7945557, 846.7377319, -604.9898682, 882.4639893, -1463.2585449, 1451.7275391
4: -362.2294006, 903.4578247, -376.6349182, 939.6113281, -1301.8406982, 1280.0926514

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_B2_A1_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2030325, upper bound: 843.2037967
time: 1.27 seconds

## Relational analysis of IS_B2_B1_B1_B2_A1_A2

### Relational analysis result of IS_B2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2030434, upper bound: 843.2038196
time: 1.01 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -199.7926483, 679.8081665, -207.7903137, 707.6581421, -907.4506836, 887.5985107
1: -325.9548950, 830.5424194, -341.5823669, 864.1693115, -1190.1240234, 1172.1246338
2: -227.5063019, 879.3715820, -236.4824371, 913.7184448, -1141.2244873, 1115.8538818
3: -579.8433838, 846.0903931, -603.6627808, 880.5806274, -1460.4238281, 1449.7528076
4: -362.1760254, 902.7009888, -375.8535767, 937.6221313, -1299.7980957, 1278.5544434

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_B2_A2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2038735, upper bound: 843.2042584
time: 0.77 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2_A2

### Relational analysis result of IS_B2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2040469, upper bound: 843.2043548
time: 0.87 seconds

## BFS IS instance: IS_B2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -204.0078278, 694.4490356, -196.5160065, 667.9773560, -871.9851685, 890.9649658
1: -333.6326904, 848.1758423, -322.7358398, 815.6779175, -1149.3104248, 1170.9114990
2: -232.3007660, 897.9718628, -223.8640594, 863.1337891, -1095.4344482, 1121.8359375
3: -592.7230835, 863.9620361, -571.4177856, 831.2780151, -1424.0010986, 1435.3796387
4: -369.6755676, 921.6985474, -356.1497192, 885.9676514, -1255.6431885, 1277.8481445

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B2_B1_B1_B1

### Relational analysis result of IS_B2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043615, upper bound: 843.2050047
time: 0.81 seconds

## Relational analysis of IS_B2_B1_B2_B1_B1_B2

### Relational analysis result of IS_B2_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2048504, upper bound: 843.2060211
time: 0.74 seconds

## BFS IS instance: IS_B2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -201.9498749, 687.5476685, -190.9664917, 649.2957764, -851.2455444, 878.5141602
1: -330.0285950, 839.7897339, -312.2252502, 793.2742310, -1123.3028564, 1152.0150146
2: -229.9509583, 889.1464844, -217.5521393, 839.8068237, -1069.7578125, 1106.6986084
3: -586.6420898, 855.4046631, -554.9062500, 808.4904785, -1395.1320801, 1410.3109131
4: -365.9924927, 912.6494751, -346.5111084, 862.2286377, -1228.2211914, 1259.1605225

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B2_B1_B2_A1

### Relational analysis result of IS_B2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2033365, upper bound: 843.2040854
time: 0.89 seconds

## Relational analysis of IS_B2_B1_B2_B1_B2_A2

### Relational analysis result of IS_B2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043401, upper bound: 843.2046206
time: 0.81 seconds

## BFS IS instance: IS_B2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -203.2297363, 691.7881470, -214.7739563, 729.7536621, -932.9832764, 906.5620117
1: -332.2830505, 844.9442749, -353.2174377, 891.3004761, -1223.5834961, 1198.1617432
2: -231.4154968, 894.5876465, -244.4272461, 942.5311279, -1173.9466553, 1139.0148926
3: -590.4360962, 860.6875000, -623.4724731, 908.1467285, -1498.5827637, 1484.1599121
4: -368.2861633, 918.2517700, -388.6141357, 966.9867554, -1335.2729492, 1306.8659668

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B2_B2_B1_B1

### Relational analysis result of IS_B2_B1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2032583, upper bound: 843.2030175
time: 1.08 seconds

## Relational analysis of IS_B2_B1_B2_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2036607, upper bound: 843.2038593
time: 0.91 seconds

## BFS IS instance: IS_B2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -201.1774139, 684.9224243, -210.8703766, 716.8565063, -918.0338135, 895.7927856
1: -328.6936340, 836.6129761, -344.9360962, 876.0980835, -1204.7917480, 1181.5490723
2: -229.0716095, 885.8025513, -239.9482727, 926.7270508, -1155.7984619, 1125.7508545
3: -584.3661499, 852.1745605, -611.1887817, 892.4113770, -1476.7773438, 1463.3632812
4: -364.6190796, 909.2324829, -381.9083557, 950.4796143, -1315.0985107, 1291.1408691

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B2_B2_B2_A1

### Relational analysis result of IS_B2_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2030440, upper bound: 843.2036409
time: 0.77 seconds

## Relational analysis of IS_B2_B1_B2_B2_B2_A2

### Relational analysis result of IS_B2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2040476, upper bound: 843.2041761
time: 1.02 seconds

## BFS IS instance: IS_B2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -203.8545532, 693.9155273, -199.4239044, 677.5343018, -881.3887329, 893.3394165
1: -333.3247070, 847.5336304, -325.7582397, 827.3570557, -1160.6817627, 1173.2918701
2: -232.1188965, 897.2911987, -226.9440765, 875.6133423, -1107.7321777, 1124.2352295
3: -592.2492065, 863.3193970, -578.4240723, 843.7814941, -1436.0307617, 1441.7434082
4: -369.3995667, 921.0283813, -361.0328064, 899.0166016, -1268.4160156, 1282.0611572

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B1_B1_B1_B1

### Relational analysis result of IS_B2_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2017624, upper bound: 843.1993435
time: 0.82 seconds

## Relational analysis of IS_B2_B2_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B1_B1_B1

### Relational analysis result of IS_B2_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2008589, upper bound: 843.1958138
time: 0.71 seconds

## Relational analysis of IS_B2_B2_B1_B1_B1_B2

### Relational analysis result of IS_B2_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2017624, upper bound: 843.1993435
time: 1.06 seconds

## BFS IS instance: IS_B2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -203.4780273, 692.6561890, -198.5858154, 674.2397461, -877.7176514, 891.2420044
1: -332.6743469, 845.9961548, -324.2622070, 823.6641846, -1156.3385010, 1170.2583008
2: -231.6951599, 895.6721802, -226.1751556, 871.8503418, -1103.5455322, 1121.8472900
3: -591.1358643, 861.7514648, -576.3778076, 840.1730957, -1431.3089600, 1438.1291504
4: -368.7308960, 919.3732300, -360.0777893, 895.4415894, -1264.1724854, 1279.4510498

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B1_B1_B2_B1

### Relational analysis result of IS_B2_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020864, upper bound: 843.2008756
time: 0.73 seconds

## Relational analysis of IS_B2_B2_B1_B1_B2_B2

### Relational analysis result of IS_B2_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020868, upper bound: 843.2006567
time: 1.08 seconds

## BFS IS instance: IS_B2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -199.0107727, 677.4154663, -214.5750122, 728.5617065, -927.5724487, 891.9904785
1: -325.3142090, 827.4483643, -351.0858459, 889.5966797, -1214.9107666, 1178.5341797
2: -226.5414886, 876.0227051, -244.0678864, 941.2116089, -1167.7528076, 1120.0903320
3: -578.1364746, 842.7551270, -621.7359619, 907.1246338, -1485.2611084, 1464.4907227
4: -360.5645752, 899.3224487, -388.0853882, 966.4195557, -1326.9841309, 1287.4075928

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B1_B2_A1_B1

### Relational analysis result of IS_B2_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2012126, upper bound: 843.2008596
time: 0.78 seconds

## Relational analysis of IS_B2_B2_B1_B2_A1_B2

### Relational analysis result of IS_B2_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2025390, upper bound: 843.2022453
time: 0.78 seconds

## BFS IS instance: IS_B2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -199.0350952, 677.2011108, -214.1457977, 727.0926514, -926.1277466, 891.3468628
1: -324.7167664, 827.3258057, -350.3522949, 887.8135986, -1212.5299072, 1177.6777344
2: -226.6563568, 876.0257568, -243.5874176, 939.3455811, -1166.0019531, 1119.6131592
3: -577.7136841, 842.8461914, -620.4862061, 905.3068848, -1483.0205078, 1463.3323975
4: -360.8432922, 899.3278198, -387.3272400, 964.5120850, -1325.3553467, 1286.6547852

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B1_B2_A2_B1

### Relational analysis result of IS_B2_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022162, upper bound: 843.2013948
time: 0.89 seconds

## Relational analysis of IS_B2_B2_B1_B2_A2_B2

### Relational analysis result of IS_B2_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2035426, upper bound: 843.2027805
time: 0.72 seconds

## BFS IS instance: IS_B2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -203.2580872, 691.8847656, -217.1502533, 738.8964844, -942.1545410, 909.0349121
1: -332.3664551, 845.0065308, -355.7048645, 902.0704956, -1234.4370117, 1200.7114258
2: -231.4593506, 894.6897583, -247.3744659, 954.3743896, -1185.8337402, 1142.0638428
3: -590.5859985, 860.7678833, -630.4425659, 920.0668945, -1510.6528320, 1491.2103271
4: -368.3508301, 918.3982544, -393.6472778, 979.5948486, -1347.9456787, 1312.0455322

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B2_B2_B1_B1_B1

### Relational analysis result of IS_B2_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2040471, upper bound: 843.2041483
time: 0.99 seconds

## Relational analysis of IS_B2_B2_B2_B1_B1_B2

### Relational analysis result of IS_B2_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2045514, upper bound: 843.2051376
time: 0.73 seconds

## BFS IS instance: IS_B2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -201.2255249, 685.0998535, -212.1663666, 721.9204102, -923.1459351, 897.2662354
1: -328.8178406, 836.7638550, -346.8035583, 881.7637939, -1210.5816650, 1183.5673828
2: -229.1423187, 886.0135498, -241.8197021, 933.5656128, -1162.7078857, 1127.8331299
3: -584.5856934, 852.3454590, -616.4452515, 899.2443848, -1483.8300781, 1468.7907715
4: -364.7298584, 909.4852295, -385.2229004, 958.6053467, -1323.3348389, 1294.7080078

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B2_B2_B1_B2_B1

### Relational analysis result of IS_B2_B2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2036389, upper bound: 843.2030492
time: 0.69 seconds

## Relational analysis of IS_B2_B2_B2_B1_B2_B2

### Relational analysis result of IS_B2_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2041887, upper bound: 843.2041929
time: 0.85 seconds

## BFS IS instance: IS_B2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -202.4320526, 689.0711670, -229.3679199, 779.4706421, -981.9026489, 918.4390869
1: -330.9474792, 841.5890503, -376.1903381, 951.6123657, -1282.5598145, 1217.7794189
2: -230.5198059, 891.1057739, -261.1895752, 1007.1514893, -1237.6711426, 1152.2952881
3: -588.1658325, 857.2958984, -665.7936401, 970.3287964, -1558.4943848, 1523.0893555
4: -366.8765869, 914.7416382, -415.4409790, 1033.9382324, -1400.8148193, 1330.1826172

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B2_B2_B2_A2_B1

### Relational analysis result of IS_B2_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1726179, upper bound: 843.2030296
time: 1.04 seconds

## Relational analysis of IS_B2_B2_B2_B2_A2_B2

### Relational analysis result of IS_B2_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1724359, upper bound: 843.2040332
time: 0.85 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.37 seconds
IS_B2_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2055856, upper bound: 843.2096253
IS_B2_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2055856, upper bound: 843.2096253
IS_B2_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2056037, upper bound: 843.2091459
IS_B2_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2056666, upper bound: 843.2088855
IS_B2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2030325, upper bound: 843.2037967
IS_B2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2030434, upper bound: 843.2038196
IS_B2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2038735, upper bound: 843.2042584
IS_B2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2040469, upper bound: 843.2043548
IS_B2_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2043615, upper bound: 843.2050047
IS_B2_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2048504, upper bound: 843.2060211
IS_B2_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2033365, upper bound: 843.2040854
IS_B2_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2043401, upper bound: 843.2046206
IS_B2_B1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2032583, upper bound: 843.2030175
IS_B2_B1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2036607, upper bound: 843.2038593
IS_B2_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2030440, upper bound: 843.2036409
IS_B2_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2040476, upper bound: 843.2041761
IS_B2_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2008589, upper bound: 843.1958138
IS_B2_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2017624, upper bound: 843.1993435
IS_B2_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2020864, upper bound: 843.2008756
IS_B2_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2020868, upper bound: 843.2006567
IS_B2_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2012126, upper bound: 843.2008596
IS_B2_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2025390, upper bound: 843.2022453
IS_B2_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2022162, upper bound: 843.2013948
IS_B2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2035426, upper bound: 843.2027805
IS_B2_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2040471, upper bound: 843.2041483
IS_B2_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2045514, upper bound: 843.2051376
IS_B2_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2036389, upper bound: 843.2030492
IS_B2_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.2041887, upper bound: 843.2041929
IS_B2_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.1726179, upper bound: 843.2030296
IS_B2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 0, lower bound: -843.1724359, upper bound: 843.2040332

## BFS IS instance: IS_B2_B1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -196.0238647, 666.6719971, -166.7506561, 566.8345337, -762.8583984, 833.4226074
1: -320.7560425, 814.4617310, -273.8630371, 692.2714233, -1013.0274658, 1088.3247070
2: -223.1674347, 861.9528198, -189.7585602, 732.1257935, -955.2932129, 1051.7114258
3: -569.3955078, 829.7605591, -484.8710327, 705.5909424, -1274.9863281, 1314.6315918
4: -355.1746216, 884.7076416, -301.9520569, 751.3421631, -1106.5168457, 1186.6595459

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B1_B1_A1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055609, upper bound: 843.2096253
time: 0.74 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_A1_B2

### Relational analysis result of IS_B2_B1_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054684, upper bound: 843.2088197
time: 0.73 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -216.6780090, 736.9144287, -166.7506561, 566.8345337, -783.5125732, 903.6649780
1: -353.8554688, 899.8789062, -273.8630371, 692.2714233, -1046.1269531, 1173.7419434
2: -246.6703949, 952.6166382, -189.7585602, 732.1257935, -978.7961426, 1142.3751221
3: -628.7636108, 917.4802246, -484.8710327, 705.5909424, -1334.3540039, 1402.3513184
4: -392.6327515, 978.0415039, -301.9520569, 751.3421631, -1143.9748535, 1279.9935303

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B1_B1_A2_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2055609, upper bound: 843.2096253
time: 1.03 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_A2_B2

### Relational analysis result of IS_B2_B1_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054684, upper bound: 843.2088204
time: 1.17 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -203.3523560, 692.3411865, -199.2061310, 676.8479614, -880.2002563, 891.5472412
1: -332.5233765, 845.5808716, -330.5152588, 825.7000122, -1158.2231445, 1176.0961914
2: -231.5299683, 895.2566528, -226.8559570, 873.2833862, -1104.8133545, 1122.1124268
3: -590.7614136, 861.2525635, -582.7919922, 841.5167847, -1432.2781982, 1444.0444336
4: -368.4442444, 918.8638916, -359.9604492, 896.4665527, -1264.9107666, 1278.8243408

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_A1

### Relational analysis result of IS_B2_B1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056037, upper bound: 843.2091459
time: 0.71 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_A2

### Relational analysis result of IS_B2_B1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056037, upper bound: 843.2091459
time: 0.72 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -202.6958008, 690.0828247, -231.1629791, 787.8473511, -990.5430908, 921.2457275
1: -331.4527283, 842.8332520, -383.1717529, 961.0306396, -1292.4833984, 1226.0048828
2: -230.7834930, 892.3457642, -263.2139282, 1016.5223389, -1247.3056641, 1155.5592041
3: -588.8859863, 858.4669800, -675.9879150, 979.0787354, -1567.9647217, 1534.4548340
4: -367.2621460, 915.8979492, -417.5289612, 1042.7260742, -1409.9882812, 1333.4268799

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_A1

### Relational analysis result of IS_B2_B1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054684, upper bound: 843.2088855
time: 0.72 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_A2

### Relational analysis result of IS_B2_B1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056666, upper bound: 843.2088855
time: 0.95 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -195.7996368, 666.5515137, -208.2344666, 709.2060547, -905.0056763, 874.7860107
1: -320.1507874, 814.2774048, -342.3541565, 866.0283203, -1186.1790771, 1156.6315918
2: -222.8350220, 861.7529907, -236.9849396, 915.6755981, -1138.5106201, 1098.7375488
3: -568.5990601, 829.2658081, -604.9898682, 882.4639893, -1451.0629883, 1434.2556152
4: -354.6743469, 884.5864868, -376.6349182, 939.6113281, -1294.2856445, 1261.2214355

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1995629, upper bound: 843.2024039
time: 0.72 seconds

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A2

### Relational analysis result of IS_B2_B1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2030325, upper bound: 843.2037967
time: 0.72 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -192.9564667, 657.4091187, -205.9428711, 701.3284912, -894.2849731, 863.3519897
1: -314.4979858, 803.3814087, -338.3788147, 856.6253662, -1171.1231689, 1141.7602539
2: -219.6997833, 850.7733154, -234.4075470, 905.7429199, -1125.4427490, 1085.1806641
3: -560.4481201, 818.1242065, -598.2203979, 872.9232178, -1433.3710938, 1416.3444824
4: -350.0313110, 873.3225708, -372.6525574, 929.4428711, -1279.4738770, 1245.9748535

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_B1

### Relational analysis result of IS_B2_B1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2024699, upper bound: 843.2027455
time: 0.72 seconds

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_B2

### Relational analysis result of IS_B2_B1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2024699, upper bound: 843.2038196
time: 0.74 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -195.5121613, 665.2845459, -207.7903137, 707.6581421, -903.1702881, 873.0748291
1: -318.9695740, 812.8703613, -341.5823669, 864.1693115, -1183.1387939, 1154.4526367
2: -222.6214600, 860.4808350, -236.4824371, 913.7184448, -1136.3398438, 1096.9632568
3: -567.3135376, 828.0255737, -603.6627808, 880.5806274, -1447.8941650, 1431.6879883
4: -354.4009399, 883.2406006, -375.8535767, 937.6221313, -1292.0228271, 1259.0939941

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A1

### Relational analysis result of IS_B2_B1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2001302, upper bound: 843.2024082
time: 0.87 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_A2

### Relational analysis result of IS_B2_B1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2038735, upper bound: 843.2042584
time: 0.72 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -195.3999176, 664.9986572, -205.4994659, 699.7839355, -895.1837769, 870.4981079
1: -318.3417969, 812.6140137, -337.6080322, 854.7772827, -1173.1190186, 1150.2220459
2: -222.5512238, 860.6837158, -233.9056244, 903.7874146, -1126.3386230, 1094.5892334
3: -567.2573242, 827.8547363, -596.8915405, 871.0509033, -1438.3082275, 1424.7459717
4: -354.5204468, 883.6094971, -371.8746033, 927.4537964, -1281.9739990, 1255.4841309

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_B1

### Relational analysis result of IS_B2_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2034561, upper bound: 843.2040799
time: 0.77 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_B2

### Relational analysis result of IS_B2_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2040469, upper bound: 843.2043548
time: 0.75 seconds

## BFS IS instance: IS_B2_B1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -204.0078278, 694.4490356, -192.6344604, 654.8911133, -858.8989258, 887.0833740
1: -333.6326904, 848.1758423, -316.4510193, 799.7734375, -1133.4061279, 1164.6267090
2: -232.3007660, 897.9718628, -219.4385834, 846.2110596, -1078.5118408, 1117.4104004
3: -592.7230835, 863.9620361, -560.2796631, 814.9467163, -1407.6693115, 1424.2416992
4: -369.6755676, 921.6985474, -349.1174011, 868.6787720, -1238.3542480, 1270.8159180

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_B1_B1_B1

### Relational analysis result of IS_B2_B1_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043615, upper bound: 843.2050047
time: 0.76 seconds

## Relational analysis of IS_B2_B1_B2_B1_B1_B1_B2

### Relational analysis result of IS_B2_B1_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2042887, upper bound: 843.2048957
time: 0.73 seconds

## BFS IS instance: IS_B2_B1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -203.6400299, 693.1874390, -192.5934448, 654.1022949, -857.7422485, 885.7807617
1: -332.9803772, 846.6359253, -315.5376587, 798.9342041, -1131.9145508, 1162.1734619
2: -231.8772583, 896.3516235, -219.4164276, 845.5800781, -1077.4572754, 1115.7680664
3: -591.6093750, 862.3949585, -559.4302979, 814.4050293, -1406.0144043, 1421.8250732
4: -369.0075684, 920.0447998, -349.1647339, 868.1186523, -1237.1262207, 1269.2094727

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_B1_B2_B1

### Relational analysis result of IS_B2_B1_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2048504, upper bound: 843.2060211
time: 0.71 seconds

## Relational analysis of IS_B2_B1_B2_B1_B1_B2_B2

### Relational analysis result of IS_B2_B1_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2048033, upper bound: 843.2058796
time: 0.96 seconds

## BFS IS instance: IS_B2_B1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -197.8695679, 673.6885376, -190.9664917, 649.2957764, -847.1652832, 864.6549683
1: -323.3812256, 822.9434204, -312.2252502, 793.2742310, -1116.6553955, 1135.1687012
2: -225.2847900, 871.2819824, -217.5521393, 839.8068237, -1065.0915527, 1088.8339844
3: -574.8908081, 838.1130371, -554.9062500, 808.4904785, -1383.3808594, 1393.0192871
4: -358.5828247, 894.3920898, -346.5111084, 862.2286377, -1220.8114014, 1240.9030762

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_B2_A1_B1

### Relational analysis result of IS_B2_B1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2026820, upper bound: 843.2008051
time: 1.00 seconds

## Relational analysis of IS_B2_B1_B2_B1_B2_A1_B2

### Relational analysis result of IS_B2_B1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2033365, upper bound: 843.2040854
time: 0.91 seconds

## BFS IS instance: IS_B2_B1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -198.0404510, 673.9130249, -190.6107178, 648.0711060, -846.1115723, 864.5237427
1: -323.0032349, 823.3414307, -311.6014404, 791.7896729, -1114.7929688, 1134.9427490
2: -225.5169220, 871.7993164, -217.1436310, 838.2327271, -1063.7496338, 1088.9429932
3: -574.7875366, 838.7404175, -553.8364258, 806.9766846, -1381.7641602, 1392.5766602
4: -359.0466003, 894.9455566, -345.8668213, 860.6198730, -1219.6665039, 1240.8123779

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_B2_A2_B1

### Relational analysis result of IS_B2_B1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2036856, upper bound: 843.2013403
time: 0.70 seconds

## Relational analysis of IS_B2_B1_B2_B1_B2_A2_B2

### Relational analysis result of IS_B2_B1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2043401, upper bound: 843.2046206
time: 0.76 seconds

## BFS IS instance: IS_B2_B1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -203.2297363, 691.7881470, -211.2218323, 717.8654175, -921.0950928, 903.0100098
1: -332.2830505, 844.9442749, -347.4396362, 876.7588501, -1209.0418701, 1192.3839111
2: -231.4154968, 894.5876465, -240.3406067, 927.0557251, -1158.4711914, 1134.9282227
3: -590.4360962, 860.6875000, -613.2671509, 893.2271729, -1483.6633301, 1473.9545898
4: -368.2861633, 918.2517700, -382.0995178, 951.2010498, -1319.4871826, 1300.3513184

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_B2_B1_B1_B1

### Relational analysis result of IS_B2_B1_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2025634, upper bound: 843.2023475
time: 0.73 seconds

## Relational analysis of IS_B2_B1_B2_B2_B1_B1_B2

### Relational analysis result of IS_B2_B1_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2032583, upper bound: 843.2030175
time: 0.76 seconds

## BFS IS instance: IS_B2_B1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -202.8687744, 690.5513916, -210.2758484, 713.8773804, -916.7460327, 900.8272095
1: -331.6428223, 843.4415283, -344.8777771, 872.2711792, -1203.9140625, 1188.3192139
2: -230.9998169, 892.9965820, -239.2919159, 922.5016479, -1153.5014648, 1132.2884521
3: -589.3395996, 859.1560669, -609.5467529, 888.7516479, -1478.0908203, 1468.7026367
4: -367.6328125, 916.6245728, -380.5755920, 946.3724365, -1314.0051270, 1297.2001953

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_B1_B2_A1

### Relational analysis result of IS_B2_B1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1975050, upper bound: 843.2009755
time: 0.74 seconds

## Relational analysis of IS_B2_B1_B2_B2_B1_B2_A2

### Relational analysis result of IS_B2_B1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2036607, upper bound: 843.2038593
time: 0.69 seconds

## BFS IS instance: IS_B2_B1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -197.0665436, 670.9613037, -210.8681641, 716.8488159, -913.9152832, 881.8293457
1: -321.9923401, 819.6134644, -344.9324341, 876.0887451, -1198.0810547, 1164.5458984
2: -224.3707886, 867.8132324, -239.9456787, 926.7172852, -1151.0881348, 1107.7587891
3: -572.5360718, 834.7254639, -611.1822510, 892.4017334, -1464.9377441, 1445.9077148
4: -357.1436462, 890.8478394, -381.9042969, 950.4695435, -1307.6131592, 1272.7520752

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_A1

### Relational analysis result of IS_B2_B1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1973715, upper bound: 843.2020427
time: 0.77 seconds

## Relational analysis of IS_B2_B1_B2_B2_B2_A1_A2

### Relational analysis result of IS_B2_B1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2030440, upper bound: 843.2036409
time: 0.73 seconds

## BFS IS instance: IS_B2_B1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -197.3227692, 671.4946899, -210.4589539, 715.4456787, -912.7684326, 881.9536133
1: -321.7825928, 820.4136963, -344.2065735, 874.3824463, -1196.1650391, 1164.6202393
2: -224.6996460, 868.7031860, -239.4754181, 924.9131470, -1149.6127930, 1108.1785889
3: -572.6949463, 835.7459106, -609.9481812, 890.6574097, -1463.3522949, 1445.6940918
4: -357.7664490, 891.7606812, -381.1571045, 948.6183472, -1306.3845215, 1272.9177246

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_A1

### Relational analysis result of IS_B2_B1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1981939, upper bound: 843.2021231
time: 0.99 seconds

## Relational analysis of IS_B2_B1_B2_B2_B2_A2_A2

### Relational analysis result of IS_B2_B1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2040476, upper bound: 843.2041761
time: 0.75 seconds

## BFS IS instance: IS_B2_B2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -203.8545532, 693.9155273, -195.2733917, 663.3956299, -867.2500610, 889.1889038
1: -333.3247070, 847.5336304, -319.0524902, 809.9348145, -1143.2595215, 1166.5861816
2: -232.1188965, 897.2911987, -222.2254181, 857.2883301, -1089.4072266, 1119.5166016
3: -592.2492065, 863.3193970, -566.3894653, 826.1330566, -1418.3823242, 1429.7088623
4: -369.3995667, 921.0283813, -353.4071960, 880.1284180, -1249.5279541, 1274.4354248

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B1_B1_B1_B1_A1

### Relational analysis result of IS_B2_B2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2006131, upper bound: 843.1957174
time: 1.13 seconds

## Relational analysis of IS_B2_B2_B1_B1_B1_B1_A2

### Relational analysis result of IS_B2_B2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2008589, upper bound: 843.1958138
time: 1.00 seconds

## BFS IS instance: IS_B2_B2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -202.1156464, 688.1217041, -191.0099640, 648.8817749, -850.9973755, 879.1316528
1: -330.3466492, 840.4429321, -312.1550293, 792.8899536, -1123.2364502, 1152.5979004
2: -230.1528931, 889.8807983, -217.5809479, 839.1112671, -1069.2641602, 1107.4617920
3: -587.1845703, 856.0862427, -554.9639893, 808.8551636, -1396.0395508, 1411.0501709
4: -366.3021851, 913.4241333, -346.5578613, 861.9239502, -1228.2260742, 1259.9819336

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B1_B1_B1_B2_A1

### Relational analysis result of IS_B2_B2_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2015889, upper bound: 843.1992319
time: 0.70 seconds

## Relational analysis of IS_B2_B2_B1_B1_B1_B2_A2

### Relational analysis result of IS_B2_B2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2017624, upper bound: 843.1993283
time: 0.74 seconds

## BFS IS instance: IS_B2_B2_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -203.4780273, 692.6561890, -194.4725189, 660.2714844, -863.7493286, 887.1287231
1: -332.6743469, 845.9961548, -317.4299927, 806.6618042, -1139.3361816, 1163.4260254
2: -231.6951599, 895.6721802, -221.4438629, 853.6033325, -1085.2984619, 1117.1160889
3: -591.1358643, 861.7514648, -564.1332397, 822.7898560, -1413.9255371, 1425.8845215
4: -368.7308960, 919.3732300, -352.5266113, 876.6110840, -1245.3420410, 1271.8999023

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B1_B2_B1_A1

### Relational analysis result of IS_B2_B2_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1962327, upper bound: 843.1987859
time: 1.13 seconds

## Relational analysis of IS_B2_B2_B1_B1_B2_B1_A2

### Relational analysis result of IS_B2_B2_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020864, upper bound: 843.2008756
time: 0.81 seconds

## BFS IS instance: IS_B2_B2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -201.4308014, 685.8209839, -191.3979950, 650.2208862, -851.6515503, 877.2189331
1: -329.1049805, 837.6738892, -311.9569092, 794.6503296, -1123.7553711, 1149.6308594
2: -229.3627777, 886.9348145, -218.1584015, 841.6035767, -1070.9663086, 1105.0931396
3: -585.1062012, 853.2472534, -556.0000610, 810.5481567, -1395.6542969, 1409.2471924
4: -365.0728455, 910.4027710, -347.7578125, 864.5427246, -1229.6154785, 1258.1606445

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B1_B2_B2_A1

### Relational analysis result of IS_B2_B2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1962331, upper bound: 843.1986441
time: 0.92 seconds

## Relational analysis of IS_B2_B2_B1_B1_B2_B2_A2

### Relational analysis result of IS_B2_B2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020868, upper bound: 843.2006567
time: 0.86 seconds

## BFS IS instance: IS_B2_B2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -199.0107727, 677.4154663, -210.3464355, 714.1281738, -913.1389160, 887.7619019
1: -325.3142090, 827.4483643, -344.1240845, 872.0070190, -1197.3210449, 1171.5725098
2: -226.5414886, 876.0227051, -239.1860809, 922.3249512, -1148.8660889, 1115.2087402
3: -578.1364746, 842.7551270, -609.1835938, 889.1657104, -1467.3016357, 1451.9384766
4: -360.5645752, 899.3224487, -380.2748108, 946.9681396, -1307.5327148, 1279.5971680

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B2_A1_B1_A1

### Relational analysis result of IS_B2_B2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1955151, upper bound: 843.1991175
time: 0.77 seconds

## Relational analysis of IS_B2_B2_B1_B2_A1_B1_A2

### Relational analysis result of IS_B2_B2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2012126, upper bound: 843.2008596
time: 0.94 seconds

## BFS IS instance: IS_B2_B2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -196.8667603, 670.3226318, -210.4406281, 715.3782349, -912.2448730, 880.7631226
1: -321.6209106, 818.8331299, -343.6756897, 873.8928833, -1195.5137939, 1162.5087891
2: -224.1438599, 866.9854736, -239.6227264, 925.0336304, -1149.1774902, 1106.6081543
3: -571.9434204, 833.9338379, -610.3056030, 891.0917969, -1463.0351562, 1444.2390137
4: -356.7970886, 890.0208130, -381.4633179, 949.6953125, -1306.4924316, 1271.4841309

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B2_A1_B2_A1

### Relational analysis result of IS_B2_B2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1968048, upper bound: 843.2006104
time: 0.75 seconds

## Relational analysis of IS_B2_B2_B1_B2_A1_B2_A2

### Relational analysis result of IS_B2_B2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2025390, upper bound: 843.2022453
time: 0.71 seconds

## BFS IS instance: IS_B2_B2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -199.0350952, 677.2011108, -209.9193878, 712.6643677, -911.6994629, 887.1204224
1: -324.7167664, 827.3258057, -343.3933105, 870.2298584, -1194.9464111, 1170.7191162
2: -226.6563568, 876.0257568, -238.7080536, 920.4671631, -1147.1235352, 1114.7337646
3: -577.7136841, 842.8461914, -607.9250488, 887.3539429, -1465.0676270, 1450.7709961
4: -360.8432922, 899.3278198, -379.5203552, 945.0692749, -1305.9123535, 1278.8480225

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B2_B1_B2_A2_B1_B1

### Relational analysis result of IS_B2_B2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020333, upper bound: 843.2012937
time: 0.75 seconds

## Relational analysis of IS_B2_B2_B1_B2_A2_B1_B2

### Relational analysis result of IS_B2_B2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022162, upper bound: 843.2013948
time: 0.70 seconds

## BFS IS instance: IS_B2_B2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -197.2415009, 671.1703491, -210.0922089, 714.1990356, -911.4405518, 881.2625732
1: -321.6654053, 820.0089722, -343.0910339, 872.4621582, -1194.1274414, 1163.0996094
2: -224.6108246, 868.2925415, -239.2339172, 923.5311890, -1148.1419678, 1107.5264893
3: -572.4935303, 835.3660889, -609.3039551, 889.6300659, -1462.1232910, 1444.6700439
4: -357.6339722, 891.3834229, -380.8503113, 948.1542969, -1305.7883301, 1272.2336426

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_B2_A2_B2_A1

### Relational analysis result of IS_B2_B2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1975499, upper bound: 843.2006121
time: 0.78 seconds

## Relational analysis of IS_B2_B2_B1_B2_A2_B2_A2

### Relational analysis result of IS_B2_B2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2035426, upper bound: 843.2027805
time: 0.96 seconds

## BFS IS instance: IS_B2_B2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -203.2580872, 691.8847656, -213.2607880, 725.6033936, -928.8614502, 905.1455078
1: -332.3664551, 845.0065308, -349.2667847, 885.8500977, -1218.2164307, 1194.2731934
2: -231.4593506, 894.6897583, -242.8766327, 937.1636963, -1168.6230469, 1137.5662842
3: -590.5859985, 860.7678833, -619.0667725, 903.4738770, -1494.0598145, 1479.8347168
4: -368.3508301, 918.3982544, -386.4795532, 962.0266113, -1330.3774414, 1304.8778076

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B2_B1_B1_B1_B1

### Relational analysis result of IS_B2_B2_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2040471, upper bound: 843.2041483
time: 1.00 seconds

## Relational analysis of IS_B2_B2_B2_B1_B1_B1_B2

### Relational analysis result of IS_B2_B2_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2040249, upper bound: 843.2041483
time: 0.92 seconds

## BFS IS instance: IS_B2_B2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -202.8921661, 690.6326904, -213.1854248, 725.2408447, -928.1329956, 903.8180542
1: -331.7195435, 843.4776001, -348.7658691, 885.6318359, -1217.3513184, 1192.2432861
2: -231.0384827, 893.0809326, -242.9875031, 937.2177734, -1168.2562256, 1136.0681152
3: -589.4793701, 859.2091675, -618.8216553, 903.2069702, -1492.6862793, 1478.0305176
4: -367.6867676, 916.7545166, -386.7905579, 962.1172485, -1329.8039551, 1303.5449219

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B2_B1_B1_B2_B1

### Relational analysis result of IS_B2_B2_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2045514, upper bound: 843.2051376
time: 0.80 seconds

## Relational analysis of IS_B2_B2_B2_B1_B1_B2_B2

### Relational analysis result of IS_B2_B2_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2045469, upper bound: 843.2051127
time: 0.74 seconds

## BFS IS instance: IS_B2_B2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -201.2255249, 685.0998535, -207.8567352, 707.5610962, -908.7866211, 892.9566040
1: -328.8178406, 836.7638550, -339.7619019, 864.2973633, -1193.1152344, 1176.5257568
2: -229.1423187, 886.0135498, -236.9228210, 914.9870605, -1144.1293945, 1122.9364014
3: -584.5856934, 852.3454590, -604.0626831, 881.3268433, -1465.9125977, 1456.4082031
4: -364.7298584, 909.4852295, -377.4839478, 939.5742188, -1304.3039551, 1286.9691162

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B2_B1_B2_B1_B1

### Relational analysis result of IS_B2_B2_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2030902, upper bound: 843.1993668
time: 0.70 seconds

## Relational analysis of IS_B2_B2_B2_B1_B2_B1_B2

### Relational analysis result of IS_B2_B2_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2036389, upper bound: 843.2030492
time: 0.71 seconds

## BFS IS instance: IS_B2_B2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -200.8692017, 683.8756104, -208.9084778, 710.0115967, -910.8806763, 892.7840576
1: -328.1889038, 835.2775879, -340.9948730, 867.4461670, -1195.6350098, 1176.2724609
2: -228.7329102, 884.4388428, -238.1014557, 918.5725708, -1147.3054199, 1122.5401611
3: -583.5059814, 850.8320923, -606.6866455, 884.6612549, -1468.1672363, 1457.5187988
4: -364.0870056, 907.8762207, -379.3770447, 943.4898071, -1307.5767822, 1287.2528076

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_B2_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B2_B1_B2_B2_B1

### Relational analysis result of IS_B2_B2_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2033730, upper bound: 843.2006089
time: 0.73 seconds

## Relational analysis of IS_B2_B2_B2_B1_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2041887, upper bound: 843.2041929
time: 1.22 seconds

## BFS IS instance: IS_B2_B2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -202.4320526, 689.0711670, -225.6863403, 767.0398560, -969.4719238, 914.7573853
1: -330.9474792, 841.5890503, -370.1052856, 936.4381714, -1267.3856201, 1211.6943359
2: -230.5198059, 891.1057739, -256.9220581, 990.9656372, -1221.4854736, 1148.0278320
3: -588.1658325, 857.2958984, -654.9591675, 954.7489014, -1542.9147949, 1512.2550049
4: -366.8765869, 914.7416382, -408.6327515, 1017.3969116, -1384.2734375, 1323.3743896

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B2_B2_A2_B1_B1

### Relational analysis result of IS_B2_B2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2034576, upper bound: 843.2030187
time: 0.98 seconds

## Relational analysis of IS_B2_B2_B2_B2_A2_B1_B2

### Relational analysis result of IS_B2_B2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2034076, upper bound: 843.2030296
time: 0.76 seconds

## BFS IS instance: IS_B2_B2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -202.0724182, 687.8400269, -224.3833466, 761.6827393, -963.7551270, 912.2233887
1: -330.3097839, 840.0936279, -367.2166748, 930.1135254, -1260.4233398, 1207.3103027
2: -230.1059113, 889.5217896, -255.5337372, 984.7493286, -1214.8552246, 1145.0552979
3: -587.0731201, 855.7716675, -650.9015503, 948.5198364, -1535.5925293, 1506.6732178
4: -366.2269592, 913.1216431, -406.5410461, 1011.2369385, -1377.4636230, 1319.6624756

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_B2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_B2_B2_B2_A2_B2_A1

### Relational analysis result of IS_B2_B2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2039153, upper bound: 843.2040331
time: 0.97 seconds

## Relational analysis of IS_B2_B2_B2_B2_A2_B2_A2

### Relational analysis result of IS_B2_B2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2039153, upper bound: 843.2040331
time: 0.75 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.78 seconds
IS_B2_B1_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2055609, upper bound: 843.2096253
IS_B2_B1_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2054684, upper bound: 843.2088197
IS_B2_B1_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2055609, upper bound: 843.2096253
IS_B2_B1_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2054684, upper bound: 843.2088204
IS_B2_B1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2056037, upper bound: 843.2091459
IS_B2_B1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2056037, upper bound: 843.2091459
IS_B2_B1_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2054684, upper bound: 843.2088855
IS_B2_B1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2056666, upper bound: 843.2088855
IS_B2_B1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.1995629, upper bound: 843.2024039
IS_B2_B1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2030325, upper bound: 843.2037967
IS_B2_B1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2024699, upper bound: 843.2027455
IS_B2_B1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2024699, upper bound: 843.2038196
IS_B2_B1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2001302, upper bound: 843.2024082
IS_B2_B1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2038735, upper bound: 843.2042584
IS_B2_B1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2034561, upper bound: 843.2040799
IS_B2_B1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2040469, upper bound: 843.2043548
IS_B2_B1_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2043615, upper bound: 843.2050047
IS_B2_B1_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2042887, upper bound: 843.2048957
IS_B2_B1_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2048504, upper bound: 843.2060211
IS_B2_B1_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2048033, upper bound: 843.2058796
IS_B2_B1_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2026820, upper bound: 843.2008051
IS_B2_B1_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2033365, upper bound: 843.2040854
IS_B2_B1_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2036856, upper bound: 843.2013403
IS_B2_B1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2043401, upper bound: 843.2046206
IS_B2_B1_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2025634, upper bound: 843.2023475
IS_B2_B1_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2032583, upper bound: 843.2030175
IS_B2_B1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.1975050, upper bound: 843.2009755
IS_B2_B1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2036607, upper bound: 843.2038593
IS_B2_B1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.1973715, upper bound: 843.2020427
IS_B2_B1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2030440, upper bound: 843.2036409
IS_B2_B1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.1981939, upper bound: 843.2021231
IS_B2_B1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2040476, upper bound: 843.2041761
IS_B2_B2_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2006131, upper bound: 843.1957174
IS_B2_B2_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2008589, upper bound: 843.1958138
IS_B2_B2_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2015889, upper bound: 843.1992319
IS_B2_B2_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2017624, upper bound: 843.1993283
IS_B2_B2_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.1962327, upper bound: 843.1987859
IS_B2_B2_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2020864, upper bound: 843.2008756
IS_B2_B2_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.1962331, upper bound: 843.1986441
IS_B2_B2_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2020868, upper bound: 843.2006567
IS_B2_B2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.1955151, upper bound: 843.1991175
IS_B2_B2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2012126, upper bound: 843.2008596
IS_B2_B2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.1968048, upper bound: 843.2006104
IS_B2_B2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2025390, upper bound: 843.2022453
IS_B2_B2_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2020333, upper bound: 843.2012937
IS_B2_B2_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2022162, upper bound: 843.2013948
IS_B2_B2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.1975499, upper bound: 843.2006121
IS_B2_B2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2035426, upper bound: 843.2027805
IS_B2_B2_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2040471, upper bound: 843.2041483
IS_B2_B2_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2040249, upper bound: 843.2041483
IS_B2_B2_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2045514, upper bound: 843.2051376
IS_B2_B2_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2045469, upper bound: 843.2051127
IS_B2_B2_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2030902, upper bound: 843.1993668
IS_B2_B2_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2036389, upper bound: 843.2030492
IS_B2_B2_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2033730, upper bound: 843.2006089
IS_B2_B2_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2041887, upper bound: 843.2041929
IS_B2_B2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2034576, upper bound: 843.2030187
IS_B2_B2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2034076, upper bound: 843.2030296
IS_B2_B2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2039153, upper bound: 843.2040331
IS_B2_B2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 0, lower bound: -843.2039153, upper bound: 843.2040331

## BFS IS instance: IS_B2_B1_B1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -196.0238647, 666.6719971, -160.9017792, 547.1442261, -743.1680908, 827.5737915
1: -320.7560425, 814.4617310, -264.6123962, 668.0552979, -988.8113403, 1079.0740967
2: -223.1674347, 861.9528198, -183.0828705, 706.5284424, -929.6958618, 1045.0356445
3: -569.3955078, 829.7605591, -468.1960754, 680.8927612, -1250.2883301, 1297.9566650
4: -355.1746216, 884.7076416, -291.1977539, 724.9754028, -1080.1500244, 1175.9050293

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B1_B1_B1_A1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2050880, upper bound: 843.2090134
time: 0.70 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_A1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054292, upper bound: 843.2091770
time: 0.82 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -195.3362122, 664.3078613, -192.4331360, 655.0974121, -850.4335938, 856.7409668
1: -319.6314392, 811.5852051, -316.9958496, 800.3209839, -1119.9523926, 1128.5809326
2: -222.3856964, 858.9022827, -218.9640350, 845.3744507, -1067.7598877, 1077.8663330
3: -567.4241943, 826.8399658, -559.7864990, 815.6773071, -1383.1013184, 1386.6264648
4: -353.9360046, 881.5969238, -347.8560791, 866.6667480, -1220.6022949, 1229.4530029

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_B1_B1_B1_A1_B2_A1

### Relational analysis result of IS_B2_B1_B1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056078, upper bound: 843.2087359
time: 0.71 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_A1_B2_A2

### Relational analysis result of IS_B2_B1_B1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2056078, upper bound: 843.2088197
time: 0.91 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -216.6780090, 736.9144287, -160.9017792, 547.1442261, -763.8222046, 897.8162231
1: -353.8554688, 899.8789062, -264.6123962, 668.0552979, -1021.9107666, 1164.4913330
2: -246.6703949, 952.6166382, -183.0828705, 706.5284424, -953.1987915, 1135.6993408
3: -628.7636108, 917.4802246, -468.1960754, 680.8927612, -1309.6563721, 1385.6762695
4: -392.6327515, 978.0415039, -291.1977539, 724.9754028, -1117.6081543, 1269.2391357

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B1_B1_B1_A2_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2049450, upper bound: 843.2089953
time: 0.76 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_A2_B1_B2

### Relational analysis result of IS_B2_B1_B1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2052863, upper bound: 843.2091624
time: 1.06 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -215.5381927, 732.9530640, -192.4331360, 655.0974121, -870.6355591, 925.3861694
1: -351.9853516, 895.0404663, -316.9958496, 800.3209839, -1152.3063965, 1212.0360107
2: -245.3635406, 947.5270386, -218.9640350, 845.3744507, -1090.7375488, 1166.4908447
3: -625.4827271, 912.5456543, -559.7864990, 815.6773071, -1441.1600342, 1472.3321533
4: -390.5563354, 972.8598022, -347.8560791, 866.6667480, -1257.2229004, 1320.7158203

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B1_B1_B1_A2_B2_A1

### Relational analysis result of IS_B2_B1_B1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2041998, upper bound: 843.2080635
time: 0.81 seconds

## Relational analysis of IS_B2_B1_B1_B1_B1_A2_B2_A2

### Relational analysis result of IS_B2_B1_B1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2052013, upper bound: 843.2086035
time: 0.99 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -194.6254578, 662.0168457, -199.2061310, 676.8479614, -871.4733887, 861.2229004
1: -318.4105530, 808.7215576, -330.5152588, 825.7000122, -1144.1103516, 1139.2368164
2: -221.5667877, 855.9904785, -226.8559570, 873.2833862, -1094.8499756, 1082.8464355
3: -565.3422852, 823.8596802, -582.7919922, 841.5167847, -1406.8591309, 1406.6516113
4: -352.6175537, 878.5537109, -359.9604492, 896.4665527, -1249.0838623, 1238.5141602

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_A1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2049522, upper bound: 843.2081909
time: 0.72 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_A1_B2

### Relational analysis result of IS_B2_B1_B1_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2053263, upper bound: 843.2088598
time: 0.96 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -214.8882141, 730.8977051, -199.2061310, 676.8479614, -891.7361450, 930.1037598
1: -350.9971924, 892.4632568, -330.5152588, 825.7000122, -1176.6968994, 1222.9785156
2: -244.6459656, 944.9099731, -226.8559570, 873.2833862, -1117.9293213, 1171.7657471
3: -623.7199707, 909.8529663, -582.7919922, 841.5167847, -1465.2368164, 1492.6450195
4: -389.4100647, 970.1484985, -359.9604492, 896.4665527, -1285.8762207, 1330.1088867

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_A2_B1

### Relational analysis result of IS_B2_B1_B1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2049522, upper bound: 843.2081909
time: 0.73 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2_B1_A2_B2

### Relational analysis result of IS_B2_B1_B1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2053263, upper bound: 843.2088598
time: 0.86 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -193.9755707, 659.7838135, -231.1629791, 787.8473511, -981.8229370, 890.9467773
1: -317.3527527, 806.0048218, -383.1717529, 961.0306396, -1278.3833008, 1189.1763916
2: -220.8283997, 853.1090088, -263.2139282, 1016.5223389, -1237.3505859, 1116.3225098
3: -563.4850464, 821.1027222, -675.9879150, 979.0787354, -1542.5637207, 1497.0905762
4: -351.4481812, 875.6182251, -417.5289612, 1042.7260742, -1394.1740723, 1293.1472168

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_A1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2049536, upper bound: 843.2078304
time: 0.76 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_A1_B2

### Relational analysis result of IS_B2_B1_B1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054181, upper bound: 843.2086370
time: 4.66 seconds

## BFS IS instance: IS_B2_B1_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -213.7403412, 726.9050903, -231.1629791, 787.8473511, -1001.5875854, 958.0680542
1: -349.1152649, 887.5875854, -383.1717529, 961.0306396, -1310.1456299, 1270.7592773
2: -243.3303223, 939.7817383, -263.2139282, 1016.5223389, -1259.8526611, 1202.9956055
3: -620.4185791, 904.8828125, -675.9879150, 979.0787354, -1599.4971924, 1580.8707275
4: -387.3187561, 964.9281616, -417.5289612, 1042.7260742, -1430.0447998, 1382.4571533

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_A2_B1

### Relational analysis result of IS_B2_B1_B1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054110, upper bound: 843.2085655
time: 1.02 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2_B2_A2_B2

### Relational analysis result of IS_B2_B1_B1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054413, upper bound: 843.2079983
time: 1.01 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -191.2874146, 651.2412109, -208.2344666, 709.2060547, -900.4934692, 859.4757080
1: -312.9464417, 795.5554810, -342.3541565, 866.0283203, -1178.9747314, 1137.9096680
2: -217.6840057, 841.7730713, -236.9849396, 915.6755981, -1133.3596191, 1078.7580566
3: -555.6619873, 810.2703857, -604.9898682, 882.4639893, -1438.1259766, 1415.2602539
4: -346.4754944, 864.2033691, -376.6349182, 939.6113281, -1286.0867920, 1240.8382568

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=901.5203857421875
rel_dist={0: [-843.2117464329822, 843.2117464329822]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -843.1904699, upper bound: 843.1932923
time: 0.71 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2116041, upper bound: 843.2116041
time: 0.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.67 seconds
IS_B1, status: Status.VERIFIED, split count: 1, time: 1.67
Output dim: 0, lower bound: -843.1904699, upper bound: 843.1932923
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -843.2116041, upper bound: 843.2116041

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -204.6976013, 696.8228149, -204.6462860, 696.6473389, -901.3449707, 901.4689941
1: -334.7795410, 851.1083374, -334.6973877, 850.8936157, -1185.6730957, 1185.8056641
2: -233.0683441, 900.9995117, -233.0101318, 900.7739868, -1133.8422852, 1134.0096436
3: -594.6608887, 866.9309082, -594.5149536, 866.7125244, -1461.3732910, 1461.4458008
4: -370.8992920, 924.7813721, -370.8064880, 924.5512695, -1295.4505615, 1295.5878906

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -843.1932923, upper bound: 843.1904699
time: 0.70 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1932923, upper bound: 843.2116041
time: 1.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.81 seconds
IS_B2_A1, status: Status.VERIFIED, split count: 2, time: 3.81
Output dim: 0, lower bound: -843.1932923, upper bound: 843.1904699
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.81
Output dim: 0, lower bound: -843.1932923, upper bound: 843.2116041

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -204.6462860, 696.6473389, -204.6462860, 696.6473389, -901.2935181, 901.2934570
1: -334.6973877, 850.8936157, -334.6973877, 850.8936157, -1185.5910645, 1185.5910645
2: -233.0101318, 900.7739868, -233.0101318, 900.7739868, -1133.7840576, 1133.7840576
3: -594.5149536, 866.7125244, -594.5149536, 866.7125244, -1461.2271729, 1461.2271729
4: -370.8064880, 924.5512695, -370.8064880, 924.5512695, -1295.3577881, 1295.3577881

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1686644, upper bound: 843.2106622
time: 0.78 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -843.1724359, upper bound: 843.1724359
time: 1.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.68 seconds
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 6.68
Output dim: 0, lower bound: -843.1686644, upper bound: 843.2106622
IS_B2_A2_A2, status: Status.VERIFIED, split count: 3, time: 6.68
Output dim: 0, lower bound: -843.1724359, upper bound: 843.1724359

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -200.7757263, 683.4359131, -204.6462860, 696.6473389, -897.4229736, 888.0821533
1: -328.3508911, 834.8297729, -334.6973877, 850.8936157, -1179.2445068, 1169.5269775
2: -228.5276184, 883.6791382, -233.0101318, 900.7739868, -1129.3012695, 1116.6892090
3: -583.2141113, 850.2399902, -594.5149536, 866.7125244, -1449.9266357, 1444.7548828
4: -363.7091370, 907.1240845, -370.8064880, 924.5512695, -1288.2603760, 1277.9305420

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2054976, upper bound: 843.2073556
time: 0.70 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2046556, upper bound: 843.2051470
time: 0.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.52 seconds
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 0, lower bound: -843.2054976, upper bound: 843.2073556
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 0, lower bound: -843.2046556, upper bound: 843.2051470

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -200.7757263, 683.4359131, -195.9719086, 666.4939575, -867.2696533, 879.4078369
1: -328.3508911, 834.8297729, -320.6729126, 814.2442627, -1142.5949707, 1155.5026855
2: -228.5276184, 883.6791382, -223.1084137, 861.7239990, -1090.2515869, 1106.7874756
3: -583.2141113, 850.2399902, -569.2477417, 829.5389404, -1412.7530518, 1419.4877930
4: -363.7091370, 907.1240845, -355.0805054, 884.4744263, -1248.1834717, 1262.2045898

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2017488, upper bound: 843.2038734
time: 0.94 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2049930, upper bound: 843.2061076
time: 0.74 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -199.0741730, 677.5504761, -216.4947510, 736.2885742, -935.3627319, 894.0452271
1: -325.4042969, 827.5818481, -353.5643005, 899.1129150, -1224.5172119, 1181.1459961
2: -226.6123657, 876.1850586, -246.4630585, 951.8201904, -1178.4323730, 1122.6478271
3: -578.3471069, 842.9053345, -628.2523193, 916.6974487, -1495.0444336, 1471.1577148
4: -360.6728821, 899.5224609, -392.3050842, 977.2288818, -1337.9017334, 1291.8273926

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2015205, upper bound: 843.2030438
time: 0.82 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2046556, upper bound: 843.2051470
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.69 seconds
IS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -843.2017488, upper bound: 843.2038734
IS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -843.2049930, upper bound: 843.2061076
IS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -843.2015205, upper bound: 843.2030438
IS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -843.2046556, upper bound: 843.2051470

## BFS IS instance: IS_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -192.8512726, 656.0115356, -195.9719086, 666.4939575, -859.3450928, 851.9834595
1: -315.6454773, 801.3678589, -320.6729126, 814.2442627, -1129.8896484, 1122.0407715
2: -219.4919281, 848.0098877, -223.1084137, 861.7239990, -1081.2159424, 1071.1182861
3: -560.2969360, 816.3878174, -569.2477417, 829.5389404, -1389.8354492, 1385.6354980
4: -349.3386230, 870.6680908, -355.0805054, 884.4744263, -1233.8127441, 1225.7485352

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
time: 0.90 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2013040, upper bound: 843.2029284
time: 0.80 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -210.9273682, 717.5479736, -194.5553284, 661.6836548, -872.6110229, 912.1032715
1: -345.9448853, 876.0031738, -318.3341064, 808.3170776, -1154.2619629, 1194.3372803
2: -240.2540283, 927.5198975, -221.5458679, 855.6128540, -1095.8666992, 1149.0657959
3: -613.3388062, 892.4967041, -565.3021851, 823.5042114, -1436.8430176, 1457.7988281
4: -382.2160950, 952.2748413, -352.5831604, 878.1868896, -1260.4029541, 1304.8580322

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2033763, upper bound: 843.2038622
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2028871, upper bound: 843.2036025
time: 0.93 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -191.0241852, 649.7196655, -216.4735260, 736.2150269, -927.2391968, 866.1930542
1: -312.4570923, 793.6718140, -353.5304260, 899.0229492, -1211.4799805, 1147.2022705
2: -217.4262085, 840.0581055, -246.4390869, 951.7265625, -1169.1525879, 1086.4967041
3: -555.0449219, 808.5714111, -628.1926880, 916.6055908, -1471.6505127, 1436.7640381
4: -346.0606689, 862.5624390, -392.2672119, 977.1333618, -1323.1938477, 1254.8295898

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2015205, upper bound: 843.2030438
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2013023, upper bound: 843.2027846
time: 0.95 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -209.5012360, 712.4988403, -215.4073486, 732.7644043, -942.2656250, 927.9060669
1: -343.5098572, 869.8354492, -351.9425049, 894.8079224, -1238.3173828, 1221.7779541
2: -238.6642303, 921.0717163, -245.2979584, 947.3641357, -1186.0283203, 1166.3696289
3: -609.2166138, 886.3721313, -625.4158936, 912.2622681, -1521.4788818, 1511.7874756
4: -379.7246399, 945.8242188, -390.5177612, 972.6993408, -1352.4239502, 1336.3419189

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2038337, upper bound: 843.2037902
time: 0.76 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2029444, upper bound: 843.2034649
time: 0.94 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.79 seconds
IS_B2_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
IS_B2_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -843.2013040, upper bound: 843.2029284
IS_B2_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -843.2033763, upper bound: 843.2038622
IS_B2_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -843.2028871, upper bound: 843.2036025
IS_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -843.2015205, upper bound: 843.2030438
IS_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -843.2013023, upper bound: 843.2027846
IS_B2_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -843.2038337, upper bound: 843.2037902
IS_B2_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -843.2029444, upper bound: 843.2034649

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -187.4764557, 637.7368164, -195.9719086, 666.4939575, -853.9703979, 833.7087402
1: -307.0308838, 778.9683838, -320.6729126, 814.2442627, -1121.2751465, 1099.6413574
2: -213.3745575, 824.1456909, -223.1084137, 861.7239990, -1075.0985107, 1047.2539062
3: -544.7655029, 793.6380615, -569.2477417, 829.5389404, -1374.3041992, 1362.8857422
4: -339.6133423, 846.2047119, -355.0805054, 884.4744263, -1224.0876465, 1201.2851562

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
time: 1.05 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
time: 0.92 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -211.8909607, 721.8567505, -194.2582550, 660.7448730, -872.6357422, 916.1149902
1: -348.0896606, 881.4781494, -317.7574463, 807.2380371, -1155.3275146, 1199.2355957
2: -241.1049805, 932.1057739, -221.1939697, 854.4157104, -1095.5206299, 1153.2996826
3: -615.5789795, 898.0225830, -564.3117065, 822.4299316, -1438.0089111, 1462.3342285
4: -383.2398682, 956.6264038, -352.0629578, 877.0118408, -1260.2517090, 1308.6888428

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2012848, upper bound: 843.2029284
time: 0.76 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2012848, upper bound: 843.2029284
time: 1.01 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -205.4995728, 699.0187378, -194.5553284, 661.6836548, -867.1832275, 893.5740967
1: -337.2778320, 853.4126587, -318.3341064, 808.3170776, -1145.5949707, 1171.7467041
2: -234.1048889, 903.4467163, -221.5458679, 855.6128540, -1089.7176514, 1124.9924316
3: -597.7308350, 869.5438843, -565.3021851, 823.5042114, -1421.2349854, 1434.8460693
4: -372.4390869, 927.5712891, -352.5831604, 878.1868896, -1250.6259766, 1280.1544189

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2030754, upper bound: 843.2038622
time: 0.78 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2030754, upper bound: 843.2038622
time: 0.78 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -223.0499420, 758.1172485, -192.9592896, 656.2207642, -879.2706299, 951.0762939
1: -366.5103149, 925.8798218, -315.5623169, 801.7031860, -1168.2135010, 1241.4421387
2: -253.8595886, 979.6130371, -219.7289734, 848.6610107, -1102.5206299, 1199.3420410
3: -647.6507568, 943.1983643, -560.5972900, 816.8030396, -1464.4538574, 1503.7956543
4: -403.6778564, 1005.2652588, -349.7388306, 871.1028442, -1274.7807617, 1355.0041504

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022745, upper bound: 843.2025823
time: 1.11 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022745, upper bound: 843.2025823
time: 0.82 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -191.0241852, 649.7196655, -211.1935272, 718.1230469, -909.1472168, 860.9132080
1: -312.4570923, 793.6718140, -345.0221558, 876.9313354, -1189.3884277, 1138.6938477
2: -217.4262085, 840.0581055, -240.4270782, 928.1463623, -1145.5725098, 1080.4849854
3: -555.0449219, 808.5714111, -612.8756104, 894.1714478, -1449.2163086, 1421.4467773
4: -346.0606689, 862.5624390, -382.6619873, 952.9562988, -1299.0169678, 1245.2243652

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2015205, upper bound: 843.2030438
time: 1.23 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2015205, upper bound: 843.2030438
time: 0.84 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -189.2911835, 643.8566284, -222.3085175, 755.4875488, -944.7787476, 866.1651611
1: -309.5284424, 786.5239258, -363.5482788, 922.4498901, -1231.9782715, 1150.0721436
2: -215.4904327, 832.6105347, -252.9343414, 976.2948608, -1191.7851562, 1085.5449219
3: -550.0663452, 801.3078003, -644.2388306, 940.3820190, -1490.4483643, 1445.5463867
4: -343.0061646, 854.9633179, -402.3085327, 1002.2784424, -1345.2845459, 1257.2717285

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007587, upper bound: 843.2017834
time: 1.04 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007587, upper bound: 843.2027846
time: 0.77 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -203.9872437, 693.6813354, -215.4073486, 732.7644043, -936.7516479, 909.0885620
1: -334.7030029, 846.8963013, -351.9425049, 894.8079224, -1229.5104980, 1198.8387451
2: -232.4150543, 896.6236572, -245.2979584, 947.3641357, -1179.7791748, 1141.9216309
3: -593.3592529, 863.0551758, -625.4158936, 912.2622681, -1505.6214600, 1488.4710693
4: -369.7919617, 920.7290039, -390.5177612, 972.6993408, -1342.4913330, 1311.2463379

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2021155, upper bound: 843.2022521
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2021155, upper bound: 843.2037902
time: 0.74 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -220.3327484, 748.5747070, -212.6237640, 723.2106323, -943.5432739, 961.1984863
1: -361.8799133, 914.2177734, -347.4214172, 883.1486206, -1245.0284424, 1261.6391602
2: -250.7904053, 967.4834595, -242.1796722, 935.1770020, -1185.9671631, 1209.6628418
3: -639.7883301, 931.4235229, -617.5429688, 900.4000854, -1540.1883545, 1548.9665527
4: -398.8571167, 993.0361328, -385.6160889, 960.3479614, -1359.2049561, 1378.6522217

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2018146, upper bound: 843.2021974
time: 0.87 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2018146, upper bound: 843.2034649
time: 0.73 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.24 seconds
IS_B2_A2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
IS_B2_A2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
IS_B2_A2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2012848, upper bound: 843.2029284
IS_B2_A2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2012848, upper bound: 843.2029284
IS_B2_A2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2030754, upper bound: 843.2038622
IS_B2_A2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2030754, upper bound: 843.2038622
IS_B2_A2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2022745, upper bound: 843.2025823
IS_B2_A2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2022745, upper bound: 843.2025823
IS_B2_A2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2015205, upper bound: 843.2030438
IS_B2_A2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2015205, upper bound: 843.2030438
IS_B2_A2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2007587, upper bound: 843.2017834
IS_B2_A2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2007587, upper bound: 843.2027846
IS_B2_A2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2021155, upper bound: 843.2022521
IS_B2_A2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2021155, upper bound: 843.2037902
IS_B2_A2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2018146, upper bound: 843.2021974
IS_B2_A2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.24
Output dim: 0, lower bound: -843.2018146, upper bound: 843.2034649

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -187.4764557, 637.7368164, -190.5143890, 647.8603516, -835.3367310, 828.2511597
1: -307.0308838, 778.9683838, -311.9286804, 791.4840698, -1098.5147705, 1090.8969727
2: -213.3745575, 824.1456909, -216.8975220, 837.4561768, -1050.8306885, 1041.0428467
3: -544.7655029, 793.6380615, -553.4931030, 806.4064941, -1351.1719971, 1347.1309814
4: -339.6133423, 846.2047119, -345.2073975, 859.6048584, -1199.2181396, 1191.4119873

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
time: 0.76 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
time: 0.97 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -187.4764557, 637.7368164, -214.9246216, 731.8286743, -919.3051147, 852.6613770
1: -307.0308838, 778.9683838, -353.0226135, 893.8590698, -1200.8898926, 1131.9908447
2: -213.3745575, 824.1456909, -244.6210480, 945.3122559, -1158.6865234, 1068.7664795
3: -544.7655029, 793.6380615, -624.3003540, 910.8497314, -1455.6151123, 1417.9384766
4: -339.6133423, 846.2047119, -388.9274597, 970.1460571, -1309.7593994, 1235.1319580

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
time: 0.86 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
time: 0.76 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -211.8909607, 721.8567505, -186.3192444, 633.2604370, -845.1513062, 908.1759033
1: -348.0896606, 881.4781494, -305.0290222, 773.7148438, -1121.8043213, 1186.5069580
2: -241.1049805, 932.1057739, -212.1348267, 818.7136841, -1059.8184814, 1144.2404785
3: -615.5789795, 898.0225830, -541.3606567, 788.4871826, -1404.0659180, 1439.3833008
4: -383.2398682, 956.6264038, -337.6451111, 840.5256958, -1223.7656250, 1294.2714844

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2019159
time: 1.01 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2029284
time: 0.78 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -211.8909607, 721.8567505, -204.2343903, 694.0287476, -905.9196167, 926.0911255
1: -348.0896606, 881.4781494, -334.9836121, 847.4652710, -1195.5548096, 1216.4617920
2: -241.1049805, 932.1057739, -232.6390076, 897.1848145, -1138.2894287, 1164.7446289
3: -615.5789795, 898.0225830, -593.7287598, 863.6718750, -1479.2508545, 1491.7513428
4: -383.2398682, 956.6264038, -370.1402588, 921.0797729, -1304.3195801, 1326.7663574

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2019159
time: 0.83 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2029284
time: 0.79 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -205.4845123, 698.9696655, -188.1023407, 639.3076782, -844.7921753, 887.0718994
1: -337.2522278, 853.3522949, -308.0525208, 781.0581665, -1118.3104248, 1161.4047852
2: -234.0876160, 903.3837280, -214.1279755, 826.3632202, -1060.4508057, 1117.5117188
3: -597.6868896, 869.4808350, -546.4844971, 795.9373779, -1393.6240234, 1415.9653320
4: -372.4120178, 927.5054932, -340.7922058, 848.3249512, -1220.7369385, 1268.2976074

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2023844, upper bound: 843.2028311
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2023844, upper bound: 843.2038622
time: 1.51 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -205.5252838, 699.1029053, -206.0993347, 700.5337524, -906.0590210, 905.2022705
1: -337.3215637, 853.5156860, -338.1727295, 855.3402710, -1192.6618652, 1191.6884766
2: -234.1344452, 903.5548096, -234.7609100, 905.4667358, -1139.6011963, 1138.3155518
3: -597.8062744, 869.6516724, -599.1926880, 871.6817627, -1469.4880371, 1468.8439941
4: -372.4855042, 927.6841431, -373.4751282, 929.4934692, -1301.9787598, 1301.1593018

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2030754, upper bound: 843.2038622
time: 1.17 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2030754, upper bound: 843.2038622
time: 0.82 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -223.0447235, 758.0991211, -188.8687744, 642.3851929, -865.4297485, 946.9678955
1: -366.5014954, 925.8581543, -308.9163818, 784.8385620, -1151.3400879, 1234.7744141
2: -253.8536987, 979.5902710, -215.0545349, 830.8103638, -1084.6640625, 1194.6447754
3: -647.6351929, 943.1761475, -548.8276978, 799.4910889, -1447.1262207, 1492.0039062
4: -403.6684875, 1005.2417603, -342.3061218, 852.8778687, -1256.5463867, 1347.5478516

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020471, upper bound: 843.2025823
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020471, upper bound: 843.2025823
time: 0.80 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -223.0553589, 758.1361084, -188.9273071, 642.1935425, -865.2487183, 947.0634155
1: -366.5193481, 925.9027710, -308.2565613, 784.7835693, -1151.3028564, 1234.1593018
2: -253.8657227, 979.6372681, -215.1450043, 830.7954712, -1084.6611328, 1194.7822266
3: -647.6666870, 943.2217407, -548.2990723, 799.6383057, -1447.3049316, 1491.5207520
4: -403.6874084, 1005.2898560, -342.5617676, 852.8497925, -1256.5372314, 1347.8515625

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022450, upper bound: 843.2025412
time: 1.09 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2022745, upper bound: 843.2036025
time: 0.78 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -184.3671112, 626.7733154, -211.1690674, 718.0388794, -902.4059448, 837.9423218
1: -301.9150085, 765.6617432, -344.9833374, 876.8288574, -1178.7438965, 1110.6450195
2: -209.7952271, 810.0073853, -240.3993988, 928.0394897, -1137.8345947, 1050.4067383
3: -535.5663452, 780.1593018, -612.8071899, 894.0666504, -1429.6326904, 1392.9665527
4: -333.9232178, 831.6394043, -382.6183167, 952.8466797, -1286.7698975, 1214.2576904

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2009886, upper bound: 843.2021421
time: 0.99 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2009886, upper bound: 843.2021421
time: 0.81 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -204.6365051, 695.4080811, -211.3331451, 718.5986328, -923.2351074, 906.7410889
1: -334.1580200, 849.2316895, -345.2428284, 877.5130005, -1211.6708984, 1194.4744873
2: -232.8652191, 898.7160645, -240.5852509, 928.7516479, -1161.6166992, 1139.3012695
3: -593.5462036, 865.9846802, -613.2625732, 894.7688599, -1488.3149414, 1479.2471924
4: -370.5267029, 922.8729248, -382.9114990, 953.5756836, -1324.1024170, 1305.7843018

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2009886, upper bound: 843.2021421
time: 0.90 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2009886, upper bound: 843.2030438
time: 0.85 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -189.2911835, 643.8566284, -218.5755463, 742.8276978, -932.1188965, 862.4321899
1: -309.5284424, 786.5239258, -357.3488464, 907.0160522, -1216.5441895, 1143.8726807
2: -215.4904327, 832.6105347, -248.5988770, 959.8116455, -1175.3018799, 1081.2092285
3: -550.0663452, 801.3078003, -633.2056885, 924.5360107, -1474.6022949, 1434.5131836
4: -343.0061646, 854.9633179, -395.3853455, 985.4236450, -1328.4298096, 1250.3486328

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2017834
time: 0.82 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2017834
time: 0.94 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -189.2911835, 643.8566284, -217.2921600, 738.2419434, -927.5331421, 861.1485596
1: -309.5284424, 786.5239258, -354.8285217, 901.7468872, -1211.2747803, 1141.3522949
2: -215.4904327, 832.6105347, -247.4057007, 954.6795044, -1170.1699219, 1080.0162354
3: -550.0663452, 801.3078003, -629.7735596, 919.3654785, -1469.4316406, 1431.0811768
4: -343.0061646, 854.9633179, -393.7373047, 980.1888428, -1323.1948242, 1248.7005615

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2027846
time: 0.82 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2027846
time: 0.82 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -203.9872437, 693.6813354, -208.4218445, 708.2461548, -912.2333984, 902.1031494
1: -334.7030029, 846.8963013, -340.4532166, 864.9156494, -1199.6182861, 1187.3494873
2: -232.4150543, 896.6236572, -237.2271576, 915.3508911, -1147.7659912, 1133.8507080
3: -593.3592529, 863.0551758, -604.6835327, 882.0658569, -1475.4249268, 1467.7387695
4: -369.7919617, 920.7290039, -377.5160522, 939.9031372, -1309.6950684, 1298.2447510

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2021155, upper bound: 843.2022521
time: 0.99 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2021155, upper bound: 843.2022521
time: 0.87 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -203.9872437, 693.6813354, -226.9131317, 772.2214355, -976.2086792, 920.5943604
1: -334.7030029, 846.8963013, -371.5665588, 942.7232666, -1277.4260254, 1218.4628906
2: -232.4150543, 896.6236572, -258.5274658, 997.7529907, -1230.1680908, 1155.1511230
3: -593.3592529, 863.0551758, -658.9298096, 961.4265747, -1554.7855225, 1521.9849854
4: -369.7919617, 920.7290039, -411.4166260, 1024.2113037, -1394.0031738, 1332.1455078

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2016337, upper bound: 843.2028003
time: 0.95 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2016337, upper bound: 843.2037902
time: 0.89 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -220.3327484, 748.5747070, -205.5541229, 698.4312134, -918.7638550, 954.1287231
1: -361.8799133, 914.2177734, -335.7739563, 852.9529419, -1214.8328857, 1249.9915771
2: -250.7904053, 967.4834595, -234.0057983, 902.8452759, -1153.6354980, 1201.4888916
3: -639.7883301, 931.4235229, -596.5635376, 869.8845825, -1509.6728516, 1527.9869385
4: -398.8571167, 993.0361328, -372.4722290, 927.2010498, -1326.0578613, 1365.5083008

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2018146, upper bound: 843.2021974
time: 0.85 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2018146, upper bound: 843.2021974
time: 0.92 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -220.3327484, 748.5747070, -223.9013062, 761.8010254, -982.1336670, 972.4758911
1: -361.8799133, 914.2177734, -366.6106262, 930.0446777, -1291.9244385, 1280.8283691
2: -250.7904053, 967.4834595, -255.1369171, 984.4552612, -1235.2452393, 1222.6202393
3: -639.7883301, 931.4235229, -650.3135986, 948.4865112, -1588.2746582, 1581.7370605
4: -398.8571167, 993.0361328, -406.0545959, 1010.6972046, -1409.5543213, 1399.0906982

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2012235, upper bound: 843.2024529
time: 1.15 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2012235, upper bound: 843.2034649
time: 0.78 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.15 seconds
IS_B2_A2_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
IS_B2_A2_A1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
IS_B2_A2_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
IS_B2_A2_A1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.1992256, upper bound: 843.2018297
IS_B2_A2_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2019159
IS_B2_A2_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2029284
IS_B2_A2_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2019159
IS_B2_A2_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2029284
IS_B2_A2_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2023844, upper bound: 843.2028311
IS_B2_A2_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2023844, upper bound: 843.2038622
IS_B2_A2_A1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2030754, upper bound: 843.2038622
IS_B2_A2_A1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2030754, upper bound: 843.2038622
IS_B2_A2_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2020471, upper bound: 843.2025823
IS_B2_A2_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2020471, upper bound: 843.2025823
IS_B2_A2_A1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2022450, upper bound: 843.2025412
IS_B2_A2_A1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2022745, upper bound: 843.2036025
IS_B2_A2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2009886, upper bound: 843.2021421
IS_B2_A2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2009886, upper bound: 843.2021421
IS_B2_A2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2009886, upper bound: 843.2021421
IS_B2_A2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2009886, upper bound: 843.2030438
IS_B2_A2_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2017834
IS_B2_A2_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2017834
IS_B2_A2_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2027846
IS_B2_A2_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2027846
IS_B2_A2_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2021155, upper bound: 843.2022521
IS_B2_A2_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2021155, upper bound: 843.2022521
IS_B2_A2_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2016337, upper bound: 843.2028003
IS_B2_A2_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2016337, upper bound: 843.2037902
IS_B2_A2_A1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2018146, upper bound: 843.2021974
IS_B2_A2_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2018146, upper bound: 843.2021974
IS_B2_A2_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2012235, upper bound: 843.2024529
IS_B2_A2_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 0, lower bound: -843.2012235, upper bound: 843.2034649

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -179.1199646, 608.9146118, -190.5143890, 647.8603516, -826.9802246, 799.4288940
1: -293.5433655, 743.7712402, -311.9286804, 791.4840698, -1085.0269775, 1055.6999512
2: -203.8157349, 786.7599487, -216.8975220, 837.4561768, -1041.2719727, 1003.6574707
3: -520.4473877, 757.9147949, -553.4931030, 806.4064941, -1326.8538818, 1311.4078369
4: -324.3930969, 807.7268677, -345.2073975, 859.6048584, -1183.9976807, 1152.9343262

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2000967, upper bound: 843.2036299
time: 1.26 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2000967, upper bound: 843.2036299
time: 0.74 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -199.2893677, 677.0875854, -190.5143890, 647.8603516, -847.1495972, 867.6019287
1: -325.5349121, 826.8152466, -311.9286804, 791.4840698, -1117.0186768, 1138.7437744
2: -226.7911682, 875.0440674, -216.8975220, 837.4561768, -1064.2473145, 1091.9414062
3: -578.0367432, 843.2169189, -553.4931030, 806.4064941, -1384.4432373, 1396.7099609
4: -360.7941284, 898.4273682, -345.2073975, 859.6048584, -1220.3989258, 1243.6347656

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2000967, upper bound: 843.2036299
time: 1.10 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2000967, upper bound: 843.2036299
time: 0.76 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -179.1199646, 608.9146118, -214.9246216, 731.8286743, -910.9486084, 823.8391724
1: -293.5433655, 743.7712402, -353.0226135, 893.8590698, -1187.4024658, 1096.7938232
2: -203.8157349, 786.7599487, -244.6210480, 945.3122559, -1149.1278076, 1031.3809814
3: -520.4473877, 757.9147949, -624.3003540, 910.8497314, -1431.2971191, 1382.2150879
4: -324.3930969, 807.7268677, -388.9274597, 970.1460571, -1294.5388184, 1196.6542969

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1990599, upper bound: 843.2018297
time: 0.77 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1990599, upper bound: 843.2018297
time: 1.04 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -199.4501648, 677.6204834, -214.9246216, 731.8286743, -931.2788086, 892.5449829
1: -325.8020020, 827.4617310, -353.0226135, 893.8590698, -1219.6611328, 1180.4842529
2: -226.9740448, 875.7231445, -244.6210480, 945.3122559, -1172.2858887, 1120.3439941
3: -578.4995117, 843.8909912, -624.3003540, 910.8497314, -1489.3489990, 1468.1914062
4: -361.0792847, 899.1301880, -388.9274597, 970.1460571, -1331.2253418, 1288.0574951

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1990599, upper bound: 843.2018297
time: 0.73 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1990599, upper bound: 843.2018297
time: 1.04 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -211.8909607, 721.8567505, -182.4983063, 620.3241577, -832.2150269, 904.3550415
1: -348.0896606, 881.4781494, -298.7423706, 757.8977661, -1105.9870605, 1180.2204590
2: -241.1049805, 932.1057739, -207.7028503, 801.9081421, -1043.0130615, 1139.8084717
3: -615.5789795, 898.0225830, -530.1746826, 772.2788696, -1387.8577881, 1428.1972656
4: -383.2398682, 956.6264038, -330.6183777, 823.3842773, -1206.6241455, 1287.2447510

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2020410
time: 0.79 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2020410
time: 1.18 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -211.8909607, 721.8567505, -182.0472717, 618.3071289, -830.1979980, 903.9040527
1: -348.0896606, 881.4781494, -297.2588806, 755.7955322, -1103.8852539, 1178.7370605
2: -241.1049805, 932.1057739, -207.3233795, 799.8278809, -1040.9328613, 1139.4289551
3: -615.5789795, 898.0225830, -528.4252319, 770.2825317, -1385.8614502, 1426.4477539
4: -383.2398682, 956.6264038, -330.0796204, 821.0907593, -1204.3305664, 1286.7060547

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2020410
time: 0.83 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2031236
time: 0.79 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -211.8909607, 721.8567505, -200.1123962, 680.1038818, -891.9948120, 921.9691162
1: -348.0896606, 881.4781494, -328.2770691, 830.4853516, -1178.5749512, 1209.7552490
2: -241.1049805, 932.1057739, -227.9332123, 879.2170410, -1120.3217773, 1160.0388184
3: -615.5789795, 898.0225830, -581.8884888, 846.2362671, -1461.8151855, 1479.9111328
4: -383.2398682, 956.6264038, -362.6547546, 902.7091064, -1285.9489746, 1319.2810059

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2019159
time: 0.85 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2019159
time: 0.80 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -211.8909607, 721.8567505, -200.4234314, 680.6907349, -892.5816650, 922.2801514
1: -348.0896606, 881.4781494, -327.9780273, 831.4168701, -1179.5064697, 1209.4560547
2: -241.1049805, 932.1057739, -228.3183594, 880.2630615, -1121.3679199, 1160.4240723
3: -615.5789795, 898.0225830, -582.0704346, 847.4720459, -1463.0507812, 1480.0930176
4: -383.2398682, 956.6264038, -363.3740845, 903.8140869, -1287.0539551, 1320.0002441

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2019159
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2_B2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2029284
time: 0.82 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -205.4837341, 698.9670410, -184.3671112, 626.7733154, -832.2568359, 883.3341064
1: -337.2508545, 853.3491211, -301.9150085, 765.6617432, -1102.9125977, 1155.2641602
2: -234.0867157, 903.3802490, -209.7952271, 810.0073853, -1044.0941162, 1113.1754150
3: -597.6845093, 869.4774780, -535.5663452, 780.1593018, -1377.8437500, 1405.0438232
4: -372.4106445, 927.5019531, -333.9232178, 831.6394043, -1204.0500488, 1261.4251709

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2023021, upper bound: 843.2028311
time: 0.97 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2021337, upper bound: 843.2027999
time: 0.84 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -205.4851379, 698.9717407, -183.7420959, 624.0469971, -829.5321045, 882.7136230
1: -337.2532349, 853.3547974, -300.1842957, 762.7668457, -1100.0200195, 1153.5389404
2: -234.0883331, 903.3862305, -209.2528229, 807.1557007, -1041.2440186, 1112.6390381
3: -597.6886597, 869.4833984, -533.4295044, 777.3768921, -1375.0654297, 1402.9128418
4: -372.4130859, 927.5081177, -333.1075745, 828.5979004, -1201.0109863, 1260.6157227

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2023021, upper bound: 843.2038622
time: 0.79 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2021337, upper bound: 843.2038516
time: 0.81 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -196.6697845, 668.5551758, -206.0993347, 700.5337524, -897.2035522, 874.6544800
1: -322.9976807, 816.3848877, -338.1727295, 855.3402710, -1178.3378906, 1154.5574951
2: -224.0425415, 864.0100708, -234.7609100, 905.4667358, -1129.5092773, 1098.7707520
3: -572.0630493, 831.9056396, -599.1926880, 871.6817627, -1443.7448730, 1431.0982666
4: -356.4426880, 887.0055542, -373.4751282, 929.4934692, -1285.9361572, 1260.4807129

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2033763, upper bound: 843.2038622
time: 1.05 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2033763, upper bound: 843.2038622
time: 0.84 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -217.7207947, 740.8000488, -206.0993347, 700.5337524, -918.2545166, 946.8994141
1: -356.6054077, 904.3693237, -338.1727295, 855.3402710, -1211.9456787, 1242.5419922
2: -247.9905853, 956.9367676, -234.7609100, 905.4667358, -1153.4572754, 1191.6976318
3: -632.2166138, 922.3754883, -599.1926880, 871.6817627, -1503.8984375, 1521.5678711
4: -394.6393738, 982.4113770, -373.4751282, 929.4934692, -1324.1326904, 1355.8864746

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2033763, upper bound: 843.2038622
time: 0.77 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2033763, upper bound: 843.2038622
time: 0.98 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -222.9808350, 757.8801880, -182.4983063, 620.3241577, -843.3048096, 940.3783569
1: -366.3932495, 925.5928345, -298.7423706, 757.8977661, -1124.2908936, 1224.3350830
2: -253.7808990, 979.3116455, -207.7028503, 801.9081421, -1055.6890869, 1187.0145264
3: -647.4463501, 942.9043579, -530.1746826, 772.2788696, -1419.7250977, 1473.0791016
4: -403.5544128, 1004.9555054, -330.6183777, 823.3842773, -1226.9387207, 1335.5738525

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020471, upper bound: 843.2025823
time: 0.79 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020471, upper bound: 843.2025823
time: 0.78 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -223.5384064, 759.8116455, -200.1123962, 680.1038818, -903.6422729, 959.9239502
1: -367.3096619, 927.9315186, -328.2770691, 830.4853516, -1197.7950439, 1256.2086182
2: -254.4109344, 981.7547607, -227.9332123, 879.2170410, -1133.6279297, 1209.6879883
3: -649.0595703, 945.2938843, -581.8884888, 846.2362671, -1495.2954102, 1527.1823730
4: -404.5406189, 1007.4479370, -362.6547546, 902.7091064, -1307.2497559, 1370.1025391

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020471, upper bound: 843.2025823
time: 0.83 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2020471, upper bound: 843.2025823
time: 0.77 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -218.9325409, 744.2448120, -188.9273071, 642.1935425, -861.1259766, 933.1721191
1: -359.8764648, 908.9308472, -308.2565613, 784.7835693, -1144.6600342, 1217.1873779
2: -249.1599121, 961.4652710, -215.1450043, 830.7954712, -1079.9552002, 1176.6102295
3: -635.7362671, 925.8845825, -548.2990723, 799.6383057, -1435.3745117, 1474.1835938
4: -396.1649780, 986.6069946, -342.5617676, 852.8497925, -1249.0147705, 1329.1687012

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2026972, upper bound: 843.2035771
time: 0.72 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2026972, upper bound: 843.2035771
time: 0.75 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -214.8932037, 731.0397339, -185.4502411, 630.6229858, -845.5160522, 916.4899902
1: -351.3764038, 893.3053589, -302.2822266, 770.7437134, -1122.1201172, 1195.5875244
2: -244.5312042, 945.2296753, -211.1818848, 815.9417725, -1060.4729004, 1156.4116211
3: -623.0174561, 909.7462769, -538.1400146, 785.2626953, -1408.2800293, 1447.8859863
4: -389.2217407, 969.6797485, -336.3494873, 837.5513306, -1226.7730713, 1306.0292969

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2027288, upper bound: 843.2036025
time: 1.14 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2027288, upper bound: 843.2036025
time: 0.92 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -184.3671112, 626.7733154, -207.3556671, 705.1308594, -889.4979248, 834.1288452
1: -301.9150085, 765.6617432, -338.6140137, 861.0618896, -1162.9769287, 1104.2756348
2: -209.7952271, 810.0073853, -235.9929810, 911.2692871, -1121.0644531, 1046.0003662
3: -535.5663452, 780.1593018, -601.5595703, 877.8920898, -1413.4582520, 1381.7188721
4: -333.9232178, 831.6394043, -375.5853271, 935.7064819, -1269.6296387, 1207.2247314

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2028581, upper bound: 843.2031578
time: 0.84 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2028581, upper bound: 843.2031578
time: 0.81 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -184.3671112, 626.7733154, -206.8480225, 703.0547485, -887.4218750, 833.6213379
1: -301.9150085, 765.6617432, -337.6954956, 858.7377930, -1160.6528320, 1103.3570557
2: -209.7952271, 810.0073853, -235.6325226, 909.2520142, -1119.0472412, 1045.6398926
3: -535.5663452, 780.1593018, -600.4957275, 875.6732178, -1411.2393799, 1380.6550293
4: -333.9232178, 831.6394043, -375.2102966, 933.8136597, -1267.7368164, 1206.8497314

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2028581, upper bound: 843.2040607
time: 0.78 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2028581, upper bound: 843.2040607
time: 0.93 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -204.6365051, 695.4080811, -207.5389252, 705.7443848, -910.3808594, 902.9470215
1: -334.1580200, 849.2316895, -338.9168701, 861.8118286, -1195.9698486, 1188.1485596
2: -232.8652191, 898.7160645, -236.2014160, 912.0521240, -1144.9173584, 1134.9172363
3: -593.5462036, 865.9846802, -602.0835571, 878.6672974, -1472.2135010, 1468.0678711
4: -370.5267029, 922.8729248, -375.9119263, 936.5152588, -1307.0415039, 1298.7845459

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1989682, upper bound: 843.1979466
time: 0.78 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1989682, upper bound: 843.2021421
time: 0.83 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -204.6365051, 695.4080811, -206.9696503, 703.4786377, -908.1151123, 902.3777466
1: -334.1580200, 849.2316895, -337.8872986, 859.2540283, -1193.4121094, 1187.1190186
2: -232.8652191, 898.7160645, -235.7689056, 909.7899780, -1142.6550293, 1134.4848633
3: -593.5462036, 865.9846802, -600.8353271, 876.1985474, -1469.7446289, 1466.8198242
4: -370.5267029, 922.8729248, -375.4239807, 934.3609619, -1304.8875732, 1298.2968750

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A2_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1989682, upper bound: 843.1994916
time: 0.84 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A2_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1989682, upper bound: 843.2021421
time: 0.79 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -182.4983063, 620.3241577, -218.5171356, 742.6326294, -925.1308594, 838.8412476
1: -298.7423706, 757.8977661, -357.2510986, 906.7803955, -1205.5227051, 1115.1489258
2: -207.7028503, 801.9081421, -248.5333252, 959.5645142, -1167.2673340, 1050.4414062
3: -530.1746826, 772.2788696, -633.0367432, 924.2937622, -1454.4685059, 1405.3153076
4: -330.6183777, 823.3842773, -395.2834473, 985.1672974, -1315.7856445, 1218.6677246

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1999460, upper bound: 843.2001055
time: 0.98 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1999460, upper bound: 843.2017834
time: 0.92 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -201.6323853, 685.1792603, -219.1206512, 744.6519165, -946.2842407, 904.2999268
1: -329.2295227, 836.7855835, -358.2549133, 909.2238770, -1238.4533691, 1195.0405273
2: -229.4771881, 885.6180420, -249.2086945, 962.1183472, -1191.5955811, 1134.8266602
3: -585.0292358, 853.2963867, -634.7751465, 926.8093872, -1511.8386230, 1488.0715332
4: -365.2536926, 909.5896606, -396.3344421, 987.8118896, -1353.0654297, 1305.9240723

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1999460, upper bound: 843.2001055
time: 0.95 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.1999460, upper bound: 843.2017834
time: 0.77 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -182.4983063, 620.3241577, -217.2321930, 738.0396729, -920.5378418, 837.5562744
1: -298.7423706, 757.8977661, -354.7268982, 901.5042725, -1200.2463379, 1112.6245117
2: -207.7028503, 801.9081421, -247.3382263, 954.4233398, -1162.1262207, 1049.2463379
3: -530.1746826, 772.2788696, -629.5989380, 919.1163940, -1449.2907715, 1401.8778076
4: -330.6183777, 823.3842773, -393.6332397, 979.9229736, -1310.5413818, 1217.0175781

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2006013, upper bound: 843.2015259
time: 1.19 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2006013, upper bound: 843.2027846
time: 0.80 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -201.6323853, 685.1792603, -217.8230286, 740.0444336, -941.6767578, 903.0022583
1: -329.2295227, 836.7855835, -355.7317200, 903.9176636, -1233.1472168, 1192.5173340
2: -229.4771881, 885.6180420, -248.0043182, 956.9620972, -1186.4393311, 1133.6223145
3: -585.0292358, 853.2963867, -631.3308105, 921.5956421, -1506.6248779, 1484.6271973
4: -365.2536926, 909.5896606, -394.6645508, 982.5546875, -1347.8083496, 1304.2541504

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2006013, upper bound: 843.2015259
time: 1.10 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2006013, upper bound: 843.2027846
time: 0.91 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -196.6697845, 668.5551758, -208.4218445, 708.2461548, -904.9159546, 876.9769897
1: -322.9976807, 816.3848877, -340.4532166, 864.9156494, -1187.9130859, 1156.8377686
2: -224.0425415, 864.0100708, -237.2271576, 915.3508911, -1139.3934326, 1101.2370605
3: -572.0630493, 831.9056396, -604.6835327, 882.0658569, -1454.1289062, 1436.5891113
4: -356.4426880, 887.0055542, -377.5160522, 939.9031372, -1296.3458252, 1264.5216064

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2018211, upper bound: 843.2008082
time: 0.84 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2018211, upper bound: 843.2022521
time: 1.04 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -217.8552551, 741.2485962, -208.4218445, 708.2461548, -926.1013794, 949.6704102
1: -356.8366089, 904.9102173, -340.4532166, 864.9156494, -1221.7521973, 1245.3634033
2: -248.1429749, 957.5144653, -237.2271576, 915.3508911, -1163.4938965, 1194.7414551
3: -632.6220703, 922.9404297, -604.6835327, 882.0658569, -1514.6879883, 1527.6236572
4: -394.8783875, 983.0167236, -377.5160522, 939.9031372, -1334.7814941, 1360.5325928

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2021155, upper bound: 843.2022346
time: 0.98 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2018966, upper bound: 843.2022521
time: 1.06 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -203.9872437, 693.6813354, -222.8862762, 758.5515137, -962.5387573, 916.5675049
1: -334.7030029, 846.8963013, -364.9239807, 926.0729980, -1260.7760010, 1211.8201904
2: -232.4150543, 896.6236572, -253.8830566, 980.0664673, -1212.4814453, 1150.5067139
3: -593.3592529, 863.0551758, -647.2227173, 944.3897705, -1537.7487793, 1510.2778320
4: -369.7919617, 920.7290039, -404.0674438, 1006.1298218, -1375.9217529, 1324.7962646

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B2_B1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2029708, upper bound: 843.2028003
time: 0.78 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B2_B1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2029708, upper bound: 843.2028003
time: 0.87 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -203.9872437, 693.6813354, -223.1149292, 758.8504639, -962.8377075, 916.7962646
1: -334.7030029, 846.8963013, -364.8377075, 926.6158447, -1261.3184814, 1211.7340088
2: -232.4150543, 896.6236572, -254.2744598, 980.9762573, -1213.3912354, 1150.8978271
3: -593.3592529, 863.0551758, -647.6643677, 944.9379272, -1538.2971191, 1510.7194824
4: -369.7919617, 920.7290039, -404.7558289, 1007.1597900, -1376.9517822, 1325.4844971

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2029708, upper bound: 843.2028003
time: 0.99 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -843.2029708, upper bound: 843.2037902
time: 1.08 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.02 seconds
IS_B2_A2_A1_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2000967, upper bound: 843.2036299
IS_B2_A2_A1_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2000967, upper bound: 843.2036299
IS_B2_A2_A1_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2000967, upper bound: 843.2036299
IS_B2_A2_A1_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2000967, upper bound: 843.2036299
IS_B2_A2_A1_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.1990599, upper bound: 843.2018297
IS_B2_A2_A1_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.1990599, upper bound: 843.2018297
IS_B2_A2_A1_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.1990599, upper bound: 843.2018297
IS_B2_A2_A1_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.1990599, upper bound: 843.2018297
IS_B2_A2_A1_B1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2020410
IS_B2_A2_A1_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2020410
IS_B2_A2_A1_B1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2020410
IS_B2_A2_A1_B1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2007050, upper bound: 843.2031236
IS_B2_A2_A1_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2019159
IS_B2_A2_A1_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2019159
IS_B2_A2_A1_B1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2019159
IS_B2_A2_A1_B1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2007569, upper bound: 843.2029284
IS_B2_A2_A1_B1_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2023021, upper bound: 843.2028311
IS_B2_A2_A1_B1_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2021337, upper bound: 843.2027999
IS_B2_A2_A1_B1_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2023021, upper bound: 843.2038622
IS_B2_A2_A1_B1_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2021337, upper bound: 843.2038516
IS_B2_A2_A1_B1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2033763, upper bound: 843.2038622
IS_B2_A2_A1_B1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2033763, upper bound: 843.2038622
IS_B2_A2_A1_B1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2033763, upper bound: 843.2038622
IS_B2_A2_A1_B1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2033763, upper bound: 843.2038622
IS_B2_A2_A1_B1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2020471, upper bound: 843.2025823
IS_B2_A2_A1_B1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2020471, upper bound: 843.2025823
IS_B2_A2_A1_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2020471, upper bound: 843.2025823
IS_B2_A2_A1_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2020471, upper bound: 843.2025823
IS_B2_A2_A1_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2026972, upper bound: 843.2035771
IS_B2_A2_A1_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2026972, upper bound: 843.2035771
IS_B2_A2_A1_B1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2027288, upper bound: 843.2036025
IS_B2_A2_A1_B1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2027288, upper bound: 843.2036025
IS_B2_A2_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2028581, upper bound: 843.2031578
IS_B2_A2_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2028581, upper bound: 843.2031578
IS_B2_A2_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2028581, upper bound: 843.2040607
IS_B2_A2_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2028581, upper bound: 843.2040607
IS_B2_A2_A1_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.1989682, upper bound: 843.1979466
IS_B2_A2_A1_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.1989682, upper bound: 843.2021421
IS_B2_A2_A1_B2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.1989682, upper bound: 843.1994916
IS_B2_A2_A1_B2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.1989682, upper bound: 843.2021421
IS_B2_A2_A1_B2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.1999460, upper bound: 843.2001055
IS_B2_A2_A1_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.1999460, upper bound: 843.2017834
IS_B2_A2_A1_B2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.1999460, upper bound: 843.2001055
IS_B2_A2_A1_B2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.1999460, upper bound: 843.2017834
IS_B2_A2_A1_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2006013, upper bound: 843.2015259
IS_B2_A2_A1_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2006013, upper bound: 843.2027846
IS_B2_A2_A1_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2006013, upper bound: 843.2015259
IS_B2_A2_A1_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2006013, upper bound: 843.2027846
IS_B2_A2_A1_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2018211, upper bound: 843.2008082
IS_B2_A2_A1_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2018211, upper bound: 843.2022521
IS_B2_A2_A1_B2_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2021155, upper bound: 843.2022346
IS_B2_A2_A1_B2_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2018966, upper bound: 843.2022521
IS_B2_A2_A1_B2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2029708, upper bound: 843.2028003
IS_B2_A2_A1_B2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2029708, upper bound: 843.2028003
IS_B2_A2_A1_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2029708, upper bound: 843.2028003
IS_B2_A2_A1_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 5.02
Output dim: 0, lower bound: -843.2029708, upper bound: 843.2037902
IS_B2_A2_A1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 5.02
Output dim: 0, lower bound: -843.2018146, upper bound: 843.2021974
IS_B2_A2_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 5.02
Output dim: 0, lower bound: -843.2018146, upper bound: 843.2021974
IS_B2_A2_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.02
Output dim: 0, lower bound: -843.2012235, upper bound: 843.2024529
IS_B2_A2_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.02
Output dim: 0, lower bound: -843.2012235, upper bound: 843.2034649
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=901.5203857421875
rel_dist={0: [-843.2116141859327, 843.2116141859326]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1115.21 seconds
