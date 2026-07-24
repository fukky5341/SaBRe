## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 1757.072574941339


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543)
1: (-485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883)
2: (-554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562)
3: (-788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867)
4: (-931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520)

## BASE Result
execution time: IAR + LP analysis = 1.22 + 2.00 = 3.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1757.1270497, upper bound: 1757.1270497


# Binary Search by BASE starts (time budget: 1196.78 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407058]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=2002.608154296875
rel_dist={0: [-1757.1264512206008, 1757.1264512206008]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=2002.608154296875
rel_dist={0: [-1757.1254498169274, 1757.1254498169274]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=2002.608154296875
rel_dist={0: [-1757.1247108341604, 1757.1247108341604]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=2002.608154296875
rel_dist={0: [-1757.1241630696206, 1757.1241630696195]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=2002.608154296875
rel_dist={0: [-1757.1238440599118, 1757.1238440599118]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=2002.608154296875
rel_dist={0: [-1757.1236760854802, 1757.1236760854804]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=2002.608154296875
rel_dist={0: [-1757.1235920351874, 1757.1235920868658]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=2002.608154296875
rel_dist={0: [-1757.1235500100406, 1757.123550071139]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=2002.608154296875
rel_dist={0: [-1757.1235290537325, 1757.1235289974684]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=2002.608154296875
rel_dist={0: [-1757.1235185379064, 1757.1235185379064]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=2002.608154296875
rel_dist={0: [-1757.123513279995, 1757.1235132380407]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=2002.608154296875
rel_dist={0: [-1757.1235106114739, 1757.1235106114736]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=2002.608154296875
rel_dist={0: [-1757.123509336572, 1757.1235093365722]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=2002.608154296875
rel_dist={0: [-1757.1235086800148, 1757.1235086800148]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=2002.608154296875
rel_dist={0: [-1757.1235083132785, 1757.1235083134302]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=2002.608154296875
rel_dist={0: [-1757.1235081491786, 1757.123508186512]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=2002.608154296875
rel_dist={0: [-1757.123508107934, 1757.1235080678216]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=2002.608154296875
rel_dist={0: [-1757.1235080660435, 1757.1235080281776]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=2002.608154296875
rel_dist={0: [-1757.1235080494814, 1757.1235080101233]}

## Binary Search Result
Binary search time: 64.74 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1132.04 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1069627, upper bound: 1757.1240331
time: 0.69 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1063020
time: 0.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.56 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.56
Output dim: 0, lower bound: -1757.1069627, upper bound: 1757.1240331
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.56
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1063020

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -395.7757874, 1606.8325195, -356.5766602, 1445.0728760, -1840.8486328, 1963.4089355
1: -485.0866699, 1794.6639404, -437.0072937, 1613.3231201, -2098.4096680, 2231.6711426
2: -554.9915771, 1824.4812012, -500.5809937, 1640.3117676, -2195.3032227, 2325.0620117
3: -788.0418701, 1987.3795166, -710.3735962, 1789.6778564, -2577.7197266, 2697.7529297
4: -931.7708130, 1855.1873779, -841.4490967, 1668.5886230, -2600.3593750, 2696.6364746

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1060141, upper bound: 1757.1060141
time: 0.58 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1060141, upper bound: 1757.1062591
time: 0.69 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -392.1190796, 1592.8139648, -457.7469177, 1887.2629395, -2279.3818359, 2050.5607910
1: -480.5942993, 1778.9794922, -559.7200317, 2107.2399902, -2587.8342285, 2338.6989746
2: -549.9753418, 1808.4610596, -649.0897827, 2136.3076172, -2686.2827148, 2457.5507812
3: -780.8599243, 1969.9682617, -917.2413330, 2344.1276855, -3124.9873047, 2887.2094727
4: -923.4982910, 1838.9149170, -1100.8392334, 2173.5678711, -3097.0661621, 2939.7541504

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1062591, upper bound: 1757.1060569
time: 0.73 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1062591, upper bound: 1757.1063020
time: 0.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.77 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -1757.1060141, upper bound: 1757.1060141
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -1757.1060141, upper bound: 1757.1062591
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -1757.1062591, upper bound: 1757.1060569
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -1757.1062591, upper bound: 1757.1063020

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -356.5766602, 1445.0728760, -356.5766602, 1445.0728760, -1801.6495361, 1801.6495361
1: -437.0072937, 1613.3231201, -437.0072937, 1613.3231201, -2050.3303223, 2050.3303223
2: -500.5809937, 1640.3117676, -500.5809937, 1640.3117676, -2140.8925781, 2140.8928223
3: -710.3735962, 1789.6778564, -710.3735962, 1789.6778564, -2500.0515137, 2500.0515137
4: -841.4490967, 1668.5886230, -841.4490967, 1668.5886230, -2510.0375977, 2510.0375977

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1066803, upper bound: 1757.1238790
time: 1.08 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1069199, upper bound: 1757.1230336
time: 0.64 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -457.7469177, 1887.2629395, -356.5766602, 1445.0728760, -1902.8198242, 2243.8393555
1: -559.7200317, 2107.2399902, -437.0072937, 1613.3231201, -2173.0427246, 2544.2473145
2: -649.0897827, 2136.3076172, -500.5809937, 1640.3117676, -2289.4016113, 2636.8886719
3: -917.2413330, 2344.1276855, -710.3735962, 1789.6778564, -2706.9191895, 3054.5012207
4: -1100.8392334, 2173.5678711, -841.4490967, 1668.5886230, -2769.4277344, 3015.0170898

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1066803, upper bound: 1757.1240331
time: 0.70 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1069199, upper bound: 1757.1232787
time: 0.67 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -356.5766602, 1445.0728760, -457.7469177, 1887.2629395, -2243.8393555, 1902.8198242
1: -437.0072937, 1613.3231201, -559.7200317, 2107.2399902, -2544.2473145, 2173.0424805
2: -500.5809937, 1640.3117676, -649.0897827, 2136.3076172, -2636.8884277, 2289.4016113
3: -710.3735962, 1789.6778564, -917.2413330, 2344.1276855, -3054.5012207, 2706.9191895
4: -841.4490967, 1668.5886230, -1100.8392334, 2173.5678711, -3015.0170898, 2769.4277344

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0404776, upper bound: 1757.1019927
time: 0.79 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1060136, upper bound: 1757.1060569
time: 0.85 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -457.7469177, 1887.2629395, -457.7469177, 1887.2629395, -2345.0095215, 2345.0095215
1: -559.7200317, 2107.2399902, -559.7200317, 2107.2399902, -2666.9597168, 2666.9597168
2: -649.0897827, 2136.3076172, -649.0897827, 2136.3076172, -2785.3974609, 2785.3974609
3: -917.2413330, 2344.1276855, -917.2413330, 2344.1276855, -3261.3691406, 3261.3691406
4: -1100.8392334, 2173.5678711, -1100.8392334, 2173.5678711, -3274.4072266, 3274.4072266

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0404771, upper bound: 1757.1019498
time: 0.76 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1060141, upper bound: 1757.1063020
time: 0.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.67 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -1757.1066803, upper bound: 1757.1238790
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -1757.1069199, upper bound: 1757.1230336
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -1757.1066803, upper bound: 1757.1240331
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -1757.1069199, upper bound: 1757.1232787
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -1757.0404776, upper bound: 1757.1019927
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -1757.1060136, upper bound: 1757.1060569
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -1757.0404771, upper bound: 1757.1019498
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -1757.1060141, upper bound: 1757.1063020

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -356.0819702, 1443.0485840, -314.7524109, 1277.9980469, -1634.0798340, 1757.8010254
1: -436.4061279, 1611.0620117, -385.6337891, 1426.8695068, -1863.2756348, 1996.6954346
2: -499.8819275, 1638.0190430, -441.0567932, 1450.1931152, -1950.0750732, 2079.0754395
3: -709.3872681, 1787.1778564, -626.3041992, 1580.1624756, -2289.5498047, 2413.4819336
4: -840.2650146, 1666.2618408, -740.3444824, 1474.6290283, -2314.8940430, 2406.6064453

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1236998
time: 0.71 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1239394
time: 0.67 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -356.5766602, 1445.0728760, -352.7528076, 1429.8717041, -1786.4483643, 1797.8256836
1: -437.0072937, 1613.3231201, -432.3590393, 1596.3819580, -2033.3892822, 2045.6818848
2: -500.5809937, 1640.3117676, -495.2670288, 1623.0344238, -2123.6154785, 2135.5786133
3: -710.3735962, 1789.6778564, -702.8384399, 1770.8116455, -2481.1850586, 2492.5161133
4: -841.4490967, 1668.5886230, -832.5136719, 1651.0700684, -2492.5190430, 2501.1022949

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239394, upper bound: 1757.1236998
time: 0.68 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239394, upper bound: 1757.1239394
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -457.3528748, 1885.6625977, -314.7524109, 1277.9980469, -1735.3509521, 2200.4150391
1: -559.2342529, 2105.4519043, -385.6337891, 1426.8695068, -1986.1037598, 2491.0856934
2: -648.5374146, 2134.4899902, -441.0567932, 1450.1931152, -2098.7304688, 2575.5468750
3: -916.4539185, 2342.1623535, -626.3041992, 1580.1624756, -2496.6162109, 2968.4663086
4: -1099.9218750, 2171.7268066, -740.3444824, 1474.6290283, -2574.5505371, 2912.0712891

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0803960, upper bound: 1757.1227017
time: 0.68 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063029, upper bound: 1757.1240331
time: 0.67 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -457.7469177, 1887.2629395, -352.7528076, 1429.8717041, -1887.6186523, 2240.0156250
1: -559.7200317, 2107.2399902, -432.3590393, 1596.3819580, -2156.1018066, 2539.5988770
2: -649.0897827, 2136.3076172, -495.2670288, 1623.0344238, -2272.1242676, 2631.5742188
3: -917.2413330, 2344.1276855, -702.8384399, 1770.8116455, -2688.0529785, 3046.9660645
4: -1100.8392334, 2173.5678711, -832.5136719, 1651.0700684, -2751.9091797, 3006.0815430

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0413834, upper bound: 1757.1189694
time: 0.66 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0413834, upper bound: 1757.1232787
time: 0.75 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -457.3528748, 1885.6625977, -2200.4150391, 1735.3509521
1: -385.6337891, 1426.8695068, -559.2342529, 2105.4519043, -2491.0856934, 1986.1037598
2: -441.0567932, 1450.1931152, -648.5374146, 2134.4899902, -2575.5468750, 2098.7304688
3: -626.3041992, 1580.1624756, -916.4539185, 2342.1623535, -2968.4665527, 2496.6162109
4: -740.3444824, 1474.6290283, -1099.9218750, 2171.7268066, -2912.0712891, 2574.5505371

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1227017, upper bound: 1757.0803960
time: 0.82 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1240331, upper bound: 1757.1063029
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -457.7469177, 1887.2629395, -2240.0156250, 1887.6186523
1: -432.3590393, 1596.3819580, -559.7200317, 2107.2399902, -2539.5988770, 2156.1018066
2: -495.2670288, 1623.0344238, -649.0897827, 2136.3076172, -2631.5744629, 2272.1242676
3: -702.8384399, 1770.8116455, -917.2413330, 2344.1276855, -3046.9660645, 2688.0529785
4: -832.5136719, 1651.0700684, -1100.8392334, 2173.5678711, -3006.0815430, 2751.9091797

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1189694, upper bound: 1757.0413833
time: 0.71 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1189694, upper bound: 1757.1069627
time: 0.69 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -457.3528748, 1885.6625977, -2279.0109863, 2075.0715332
1: -481.2612000, 1806.1429443, -559.2342529, 2105.4519043, -2586.7131348, 2365.3769531
2: -557.5192261, 1832.2569580, -648.5374146, 2134.4899902, -2692.0092773, 2480.7944336
3: -787.0695801, 2007.7425537, -916.4539185, 2342.1623535, -3129.2319336, 2924.1960449
4: -943.0974121, 1864.2763672, -1099.9218750, 2171.7268066, -3114.8239746, 2964.1982422

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0364133, upper bound: 1757.0364133
time: 0.59 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0364134, upper bound: 1757.1019499
time: 0.73 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -457.7469177, 1887.2629395, -2337.4558105, 2314.3166504
1: -550.5508423, 2073.0170898, -559.7200317, 2107.2399902, -2657.7907715, 2632.7363281
2: -638.3911133, 2101.5507812, -649.0897827, 2136.3076172, -2774.6987305, 2750.6406250
3: -902.1766968, 2305.8959961, -917.2413330, 2344.1276855, -3246.3044434, 3223.1372070
4: -1082.5723877, 2138.3459473, -1100.8392334, 2173.5678711, -3256.1396484, 3239.1850586

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1019926, upper bound: 1757.0407226
time: 0.67 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1019926, upper bound: 1757.1063020
time: 0.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.68 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1236998
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1239394
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1239394, upper bound: 1757.1236998
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1239394, upper bound: 1757.1239394
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.0803960, upper bound: 1757.1227017
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1063029, upper bound: 1757.1240331
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.0413834, upper bound: 1757.1189694
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.0413834, upper bound: 1757.1232787
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1227017, upper bound: 1757.0803960
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1240331, upper bound: 1757.1063029
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1189694, upper bound: 1757.0413833
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1189694, upper bound: 1757.1069627
IS_B2_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.0364133, upper bound: 1757.0364133
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.0364134, upper bound: 1757.1019499
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1019926, upper bound: 1757.0407226
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -1757.1019926, upper bound: 1757.1063020

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -314.7524109, 1277.9980469, -1592.7503662, 1592.7503662
1: -385.6337891, 1426.8695068, -385.6337891, 1426.8695068, -1812.5030518, 1812.5030518
2: -441.0567932, 1450.1931152, -441.0567932, 1450.1931152, -1891.2498779, 1891.2498779
3: -626.3041992, 1580.1624756, -626.3041992, 1580.1624756, -2206.4663086, 2206.4663086
4: -740.3444824, 1474.6290283, -740.3444824, 1474.6290283, -2214.9733887, 2214.9731445

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_B1
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
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0716854, upper bound: 1757.1183511
time: 0.64 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1245527
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -314.7524109, 1277.9980469, -1630.7508545, 1744.6241455
1: -432.3590393, 1596.3819580, -385.6337891, 1426.8695068, -1859.2283936, 1982.0156250
2: -495.2670288, 1623.0344238, -441.0567932, 1450.1931152, -1945.4600830, 2064.0913086
3: -702.8384399, 1770.8116455, -626.3041992, 1580.1624756, -2283.0007324, 2397.1157227
4: -832.5136719, 1651.0700684, -740.3444824, 1474.6290283, -2307.1425781, 2391.4145508

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1213372, upper bound: 1757.1247288
time: 0.85 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1247923
time: 0.69 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -352.7528076, 1429.8717041, -1744.6240234, 1630.7508545
1: -385.6337891, 1426.8695068, -432.3590393, 1596.3819580, -1982.0155029, 1859.2283936
2: -441.0567932, 1450.1931152, -495.2670288, 1623.0344238, -2064.0913086, 1945.4600830
3: -626.3041992, 1580.1624756, -702.8384399, 1770.8116455, -2397.1157227, 2283.0007324
4: -740.3444824, 1474.6290283, -832.5136719, 1651.0700684, -2391.4145508, 2307.1425781

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0716854, upper bound: 1757.1174983
time: 0.67 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1236998
time: 0.75 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -352.7528076, 1429.8717041, -1782.6245117, 1782.6245117
1: -432.3590393, 1596.3819580, -432.3590393, 1596.3819580, -2028.7409668, 2028.7409668
2: -495.2670288, 1623.0344238, -495.2670288, 1623.0344238, -2118.3015137, 2118.3015137
3: -702.8384399, 1770.8116455, -702.8384399, 1770.8116455, -2473.6499023, 2473.6499023
4: -832.5136719, 1651.0700684, -832.5136719, 1651.0700684, -2483.5837402, 2483.5837402

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0716854, upper bound: 1757.1174983
time: 0.72 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1239394
time: 0.74 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -466.1533203, 1924.0361328, -312.8979492, 1270.4250488, -1736.5783691, 2236.9340820
1: -569.0003662, 2148.2441406, -383.3760681, 1418.4169922, -1987.4173584, 2531.6201172
2: -661.5056152, 2176.6816406, -438.4498291, 1441.6322021, -2103.1376953, 2615.1313477
3: -934.1921997, 2393.3369141, -622.5888672, 1570.7271729, -2504.9191895, 3015.9252930
4: -1125.7482910, 2214.2114258, -735.8646851, 1465.9256592, -2591.6735840, 2950.0761719

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9453307, upper bound: 1757.1059715
time: 0.67 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9453307, upper bound: 1757.1227017
time: 0.72 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -455.0896912, 1876.6630859, -314.7524109, 1277.9980469, -1733.0877686, 2191.4152832
1: -556.4686279, 2095.4030762, -385.6337891, 1426.8695068, -1983.3381348, 2481.0368652
2: -645.3818970, 2124.2795410, -441.0567932, 1450.1931152, -2095.5747070, 2565.3364258
3: -911.9818726, 2331.0532227, -626.3041992, 1580.1624756, -2492.1442871, 2957.3574219
4: -1094.6657715, 2161.3532715, -740.3444824, 1474.6290283, -2569.2944336, 2901.6977539

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1198479
time: 0.70 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1240331
time: 0.93 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -352.7528076, 1429.8717041, -1823.2200928, 1970.4714355
1: -481.2612000, 1806.1429443, -432.3590393, 1596.3819580, -2077.6430664, 2238.5019531
2: -557.5192261, 1832.2569580, -495.2670288, 1623.0344238, -2180.5537109, 2327.5239258
3: -787.0695801, 2007.7425537, -702.8384399, 1770.8116455, -2557.8813477, 2710.5805664
4: -943.0974121, 1864.2763672, -832.5136719, 1651.0700684, -2594.1669922, 2696.7900391

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_A1

### Relational analysis result of IS_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9453307, upper bound: 1757.0978760
time: 0.63 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2

### Relational analysis result of IS_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1189676
time: 0.73 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -352.7528076, 1429.8717041, -1880.0645752, 2209.3227539
1: -550.5508423, 2073.0170898, -432.3590393, 1596.3819580, -2146.9328613, 2505.3757324
2: -638.3911133, 2101.5507812, -495.2670288, 1623.0344238, -2261.4255371, 2596.8178711
3: -902.1766968, 2305.8959961, -702.8384399, 1770.8116455, -2672.9882812, 3008.7343750
4: -1082.5723877, 2138.3459473, -832.5136719, 1651.0700684, -2733.6418457, 2970.8596191

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9453307, upper bound: 1757.1222227
time: 0.78 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1232769
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -312.8979492, 1270.4250488, -466.1533203, 1924.0361328, -2236.9340820, 1736.5783691
1: -383.3760681, 1418.4169922, -569.0003662, 2148.2441406, -2531.6201172, 1987.4173584
2: -438.4498291, 1441.6322021, -661.5056152, 2176.6816406, -2615.1313477, 2103.1376953
3: -622.5888672, 1570.7271729, -934.1921997, 2393.3369141, -3015.9252930, 2504.9194336
4: -735.8646851, 1465.9256592, -1125.7482910, 2214.2114258, -2950.0761719, 2591.6735840

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_A1_A1_B1_B1

### Relational analysis result of IS_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059715, upper bound: 1756.9453307
time: 1.05 seconds

## Relational analysis of IS_B2_A1_A1_B1_B2

### Relational analysis result of IS_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059715, upper bound: 1757.0803960
time: 0.72 seconds

## BFS IS instance: IS_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -455.0896912, 1876.6630859, -2191.4155273, 1733.0877686
1: -385.6337891, 1426.8695068, -556.4686279, 2095.4030762, -2481.0368652, 1983.3381348
2: -441.0567932, 1450.1931152, -645.3818970, 2124.2795410, -2565.3364258, 2095.5749512
3: -626.3041992, 1580.1624756, -911.9818726, 2331.0532227, -2957.3571777, 2492.1442871
4: -740.3444824, 1474.6290283, -1094.6657715, 2161.3532715, -2901.6977539, 2569.2944336

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B2_A1_A1_B2_B1

### Relational analysis result of IS_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198479, upper bound: 1757.0393753
time: 0.68 seconds

## Relational analysis of IS_B2_A1_A1_B2_B2

### Relational analysis result of IS_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198479, upper bound: 1757.1063029
time: 0.91 seconds

## BFS IS instance: IS_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -393.3483887, 1617.7186279, -1970.4714355, 1823.2200928
1: -432.3590393, 1596.3819580, -481.2612000, 1806.1429443, -2238.5019531, 2077.6430664
2: -495.2670288, 1623.0344238, -557.5192261, 1832.2569580, -2327.5239258, 2180.5537109
3: -702.8384399, 1770.8116455, -787.0695801, 2007.7425537, -2710.5805664, 2557.8813477
4: -832.5136719, 1651.0700684, -943.0974121, 1864.2763672, -2696.7900391, 2594.1669922

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A2_B1_B1

### Relational analysis result of IS_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0978760, upper bound: 1756.9453972
time: 0.65 seconds

## Relational analysis of IS_B2_A1_A2_B1_B2

### Relational analysis result of IS_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1189675, upper bound: 1757.0396149
time: 0.89 seconds

## BFS IS instance: IS_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -450.1930237, 1856.5698242, -2209.3227539, 1880.0646973
1: -432.3590393, 1596.3819580, -550.5508423, 2073.0170898, -2505.3759766, 2146.9328613
2: -495.2670288, 1623.0344238, -638.3911133, 2101.5507812, -2596.8178711, 2261.4255371
3: -702.8384399, 1770.8116455, -902.1766968, 2305.8959961, -3008.7343750, 2672.9882812
4: -832.5136719, 1651.0700684, -1082.5723877, 2138.3459473, -2970.8596191, 2733.6418457

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A2_B2_B1

### Relational analysis result of IS_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0978760, upper bound: 1757.0808672
time: 0.79 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2

### Relational analysis result of IS_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1189675, upper bound: 1757.1054504
time: 0.74 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -450.1930237, 1856.5698242, -2249.9177246, 2067.9116211
1: -481.2612000, 1806.1429443, -550.5508423, 2073.0170898, -2554.2778320, 2356.6938477
2: -557.5192261, 1832.2569580, -638.3911133, 2101.5507812, -2659.0700684, 2470.6479492
3: -787.0695801, 2007.7425537, -902.1766968, 2305.8959961, -3092.9655762, 2909.9189453
4: -943.0974121, 1864.2763672, -1082.5723877, 2138.3459473, -3081.4428711, 2946.8481445

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9382658, upper bound: 1757.0736512
time: 0.70 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0346449, upper bound: 1757.1015369
time: 0.65 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -393.3483887, 1617.7186279, -2067.9116211, 2249.9179688
1: -550.5508423, 2073.0170898, -481.2612000, 1806.1429443, -2356.6936035, 2554.2780762
2: -638.3911133, 2101.5507812, -557.5192261, 1832.2569580, -2470.6479492, 2659.0700684
3: -902.1766968, 2305.8959961, -787.0695801, 2007.7425537, -2909.9189453, 3092.9655762
4: -1082.5723877, 2138.3459473, -943.0974121, 1864.2763672, -2946.8481445, 3081.4428711

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0768801, upper bound: 1757.0379205
time: 0.78 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015724, upper bound: 1757.0389542
time: 0.95 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -450.1930237, 1856.5698242, -2306.7626953, 2306.7626953
1: -550.5508423, 2073.0170898, -550.5508423, 2073.0170898, -2623.5678711, 2623.5678711
2: -638.3911133, 2101.5507812, -638.3911133, 2101.5507812, -2739.9418945, 2739.9418945
3: -902.1766968, 2305.8959961, -902.1766968, 2305.8959961, -3208.0727539, 3208.0727539
4: -1082.5723877, 2138.3459473, -1082.5723877, 2138.3459473, -3220.9177246, 3220.9177246

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0768800, upper bound: 1757.1048669
time: 0.73 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015724, upper bound: 1757.1056548
time: 0.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.78 seconds
IS_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.0716854, upper bound: 1757.1183511
IS_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1245527
IS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1213372, upper bound: 1757.1247288
IS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1247923
IS_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.0716854, upper bound: 1757.1174983
IS_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1236998
IS_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.0716854, upper bound: 1757.1174983
IS_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1236998, upper bound: 1757.1239394
IS_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1756.9453307, upper bound: 1757.1059715
IS_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1756.9453307, upper bound: 1757.1227017
IS_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1198479
IS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1240331
IS_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1756.9453307, upper bound: 1757.0978760
IS_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1189676
IS_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1756.9453307, upper bound: 1757.1222227
IS_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1232769
IS_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1059715, upper bound: 1756.9453307
IS_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1059715, upper bound: 1757.0803960
IS_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1198479, upper bound: 1757.0393753
IS_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1198479, upper bound: 1757.1063029
IS_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.0978760, upper bound: 1756.9453972
IS_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1189675, upper bound: 1757.0396149
IS_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.0978760, upper bound: 1757.0808672
IS_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1189675, upper bound: 1757.1054504
IS_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1756.9382658, upper bound: 1757.0736512
IS_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.0346449, upper bound: 1757.1015369
IS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.0768801, upper bound: 1757.0379205
IS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1015724, upper bound: 1757.0389542
IS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.0768800, upper bound: 1757.1048669
IS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.78
Output dim: 0, lower bound: -1757.1015724, upper bound: 1757.1056548

## BFS IS instance: IS_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -303.3241882, 1234.4622803, -313.3843689, 1272.5599365, -1575.8841553, 1547.8466797
1: -371.4692993, 1378.0418701, -383.9473267, 1420.7854004, -1792.2545166, 1761.9888916
2: -425.6004028, 1400.6689453, -439.1740417, 1444.0181885, -1869.6186523, 1839.8430176
3: -603.7112427, 1526.6672363, -623.5877075, 1573.4835205, -2177.1943359, 2150.2546387
4: -714.8023071, 1424.3453369, -737.2165527, 1468.3677979, -2183.1699219, 2161.5620117

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1076380, upper bound: 1757.1216996
time: 0.60 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244855, upper bound: 1757.1221901
time: 0.71 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -313.4305115, 1272.7999268, -314.7524109, 1277.9980469, -1591.4285889, 1587.5522461
1: -384.0217285, 1421.0507812, -385.6337891, 1426.8695068, -1810.8912354, 1806.6843262
2: -439.1771240, 1444.3166504, -441.0567932, 1450.1931152, -1889.3702393, 1885.3732910
3: -623.7006836, 1573.6455078, -626.3041992, 1580.1624756, -2203.8632812, 2199.9492188
4: -737.1726074, 1468.6575928, -740.3444824, 1474.6290283, -2211.8015137, 2209.0017090

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1226702, upper bound: 1757.1082130
time: 0.68 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1245527, upper bound: 1757.1245527
time: 0.64 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -351.6362305, 1425.4775391, -303.3241882, 1234.4622803, -1586.0982666, 1728.8017578
1: -430.9883728, 1591.4641113, -371.4692993, 1378.0418701, -1809.0301514, 1962.9332275
2: -493.7317810, 1618.0450439, -425.6004028, 1400.6689453, -1894.4006348, 2043.6455078
3: -700.6330566, 1765.4210205, -603.7112427, 1526.6672363, -2227.3002930, 2369.1320801
4: -829.9680176, 1646.0084229, -714.8023071, 1424.3453369, -2254.3134766, 2360.8105469

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1145709, upper bound: 1757.1240235
time: 0.74 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1159619, upper bound: 1757.1240035
time: 0.72 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -313.4305115, 1272.7999268, -1625.5527344, 1743.3022461
1: -432.3590393, 1596.3819580, -384.0217285, 1421.0507812, -1853.4096680, 1980.4036865
2: -495.2670288, 1623.0344238, -439.1771240, 1444.3166504, -1939.5836182, 2062.2114258
3: -702.8384399, 1770.8116455, -623.7006836, 1573.6455078, -2276.4836426, 2394.5122070
4: -832.5136719, 1651.0700684, -737.1726074, 1468.6575928, -2301.1713867, 2388.2426758

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1169335, upper bound: 1757.1238128
time: 0.66 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1183245, upper bound: 1757.1237928
time: 0.72 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -303.3241882, 1234.4622803, -351.6362305, 1425.4775391, -1728.8017578, 1586.0982666
1: -371.4692993, 1378.0418701, -430.9883728, 1591.4641113, -1962.9332275, 1809.0301514
2: -425.6004028, 1400.6689453, -493.7317810, 1618.0450439, -2043.6455078, 1894.4007568
3: -603.7112427, 1526.6672363, -700.6330566, 1765.4210205, -2369.1320801, 2227.3002930
4: -714.8023071, 1424.3453369, -829.9680176, 1646.0084229, -2360.8105469, 2254.3134766

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A1_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1240235, upper bound: 1757.1145709
time: 0.68 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1240035, upper bound: 1757.1159619
time: 0.73 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -313.4305115, 1272.7999268, -352.7528076, 1429.8717041, -1743.3022461, 1625.5527344
1: -384.0217285, 1421.0507812, -432.3590393, 1596.3819580, -1980.4036865, 1853.4096680
2: -439.1771240, 1444.3166504, -495.2670288, 1623.0344238, -2062.2114258, 1939.5836182
3: -623.7006836, 1573.6455078, -702.8384399, 1770.8116455, -2394.5122070, 2276.4836426
4: -737.1726074, 1468.6575928, -832.5136719, 1651.0700684, -2388.2426758, 2301.1713867

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A1_A2_B1

### Relational analysis result of IS_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1238128, upper bound: 1757.1169335
time: 0.82 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_B2

### Relational analysis result of IS_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237928, upper bound: 1757.1183245
time: 0.71 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -342.4194946, 1390.7973633, -351.6362305, 1425.4775391, -1767.8969727, 1742.4334717
1: -419.5917969, 1552.5744629, -430.9883728, 1591.4641113, -2011.0559082, 1983.5628662
2: -481.1055603, 1578.5659180, -493.7317810, 1618.0450439, -2099.1506348, 2072.2976074
3: -682.4114380, 1722.1546631, -700.6330566, 1765.4210205, -2447.8322754, 2422.7873535
4: -808.9281616, 1605.6789551, -829.9680176, 1646.0084229, -2454.9360352, 2435.6469727

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0647506, upper bound: 1757.1173157
time: 0.75 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B2_A2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0620871, upper bound: 1757.1051170
time: 0.73 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_A2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0656714, upper bound: 1757.1171402
time: 0.69 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A1_B2_A2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0654839, upper bound: 1757.0654839
time: 0.69 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0654839, upper bound: 1757.1174983
time: 0.69 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -350.8810730, 1422.3142090, -352.7528076, 1429.8717041, -1780.7528076, 1775.0670166
1: -430.0623474, 1587.9256592, -432.3590393, 1596.3819580, -2026.4443359, 2020.2844238
2: -492.6302185, 1614.4853516, -495.2670288, 1623.0344238, -2115.6645508, 2109.7524414
3: -699.1375732, 1761.4387207, -702.8384399, 1770.8116455, -2469.9489746, 2464.2770996
4: -828.0973511, 1642.4067383, -832.5136719, 1651.0700684, -2479.1674805, 2474.9204102

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1171477, upper bound: 1757.1185641
time: 0.77 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1183245, upper bound: 1757.1185441
time: 0.71 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -394.0575562, 1624.7408447, -312.8979492, 1270.4250488, -1664.4826660, 1937.6387939
1: -481.0112000, 1813.6910400, -383.3760681, 1418.4169922, -1899.4279785, 2197.0671387
2: -559.5606079, 1838.4639893, -438.4498291, 1441.6322021, -2001.1927490, 2276.9133301
3: -788.6979370, 2019.8027344, -622.5888672, 1570.7271729, -2359.4250488, 2642.3911133
4: -950.7693481, 1869.5782471, -735.8646851, 1465.9256592, -2416.6948242, 2605.4428711

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B1_A1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9420540, upper bound: 1757.1059715
time: 0.68 seconds

## Relational analysis of IS_B1_A2_B1_A1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9453307, upper bound: 1757.1055435
time: 0.71 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -458.6496277, 1893.6781006, -312.8979492, 1270.4250488, -1729.0745850, 2206.5761719
1: -559.9286499, 2114.3937988, -383.3760681, 1418.4169922, -1978.3453369, 2496.8854980
2: -650.8837891, 2142.2656250, -438.4498291, 1441.6322021, -2092.5161133, 2580.7150879
3: -919.2867432, 2355.4443359, -622.5888672, 1570.7271729, -2490.0139160, 2978.0329590
4: -1107.5815430, 2179.3068848, -735.8646851, 1465.9256592, -2573.5070801, 2915.1716309

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B1_A1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9420540, upper bound: 1757.1217634
time: 0.65 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9453307, upper bound: 1757.1227017
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -391.1911621, 1608.9113770, -314.7524109, 1277.9980469, -1669.1890869, 1923.6636963
1: -478.6294861, 1796.3077393, -385.6337891, 1426.8695068, -1905.4986572, 2181.9414062
2: -554.5004883, 1822.2730713, -441.0567932, 1450.1931152, -2004.6936035, 2263.3298340
3: -782.7835693, 1996.8884277, -626.3041992, 1580.1624756, -2362.9460449, 2623.1926270
4: -938.0429688, 1854.0861816, -740.3444824, 1474.6290283, -2412.6711426, 2594.4306641

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0370100, upper bound: 1757.1197588
time: 0.62 seconds

## Relational analysis of IS_B1_A2_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1198223
time: 0.76 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -447.9715271, 1847.6967773, -314.7524109, 1277.9980469, -1725.9696045, 2162.4489746
1: -547.8370361, 2063.1066895, -385.6337891, 1426.8695068, -1974.7064209, 2448.7402344
2: -635.2952881, 2091.4885254, -441.0567932, 1450.1931152, -2085.4882812, 2532.5454102
3: -897.7854614, 2294.9624023, -626.3041992, 1580.1624756, -2477.9475098, 2921.2661133
4: -1077.4190674, 2128.1311035, -740.3444824, 1474.6290283, -2552.0478516, 2868.4755859

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0370100, upper bound: 1757.1234787
time: 0.71 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1240331
time: 0.62 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -394.0575562, 1624.7408447, -350.6282654, 1421.1749268, -1815.2324219, 1975.3691406
1: -481.0112000, 1813.6910400, -429.7668762, 1586.6865234, -2067.6977539, 2243.4580078
2: -559.5606079, 1838.4639893, -492.2691956, 1613.2059326, -2172.7663574, 2330.7331543
3: -788.6979370, 2019.8027344, -698.5472412, 1759.9062500, -2548.6042480, 2718.3500977
4: -950.7693481, 1869.5782471, -827.3372803, 1641.0731201, -2591.8422852, 2696.9155273

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9283175, upper bound: 1757.0851329
time: 0.90 seconds

## Relational analysis of IS_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9283175, upper bound: 1757.0870045
time: 0.70 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -391.1911621, 1608.9113770, -352.7528076, 1429.8717041, -1821.0628662, 1961.6641846
1: -478.6294861, 1796.3077393, -432.3590393, 1596.3819580, -2075.0114746, 2228.6665039
2: -554.5004883, 1822.2730713, -495.2670288, 1623.0344238, -2177.5349121, 2317.5400391
3: -782.7835693, 1996.8884277, -702.8384399, 1770.8116455, -2553.5952148, 2699.7268066
4: -938.0429688, 1854.0861816, -832.5136719, 1651.0700684, -2589.1125488, 2686.5998535

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0262657, upper bound: 1757.1116861
time: 0.73 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0262457, upper bound: 1757.1132523
time: 0.68 seconds

## BFS IS instance: IS_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -458.6496277, 1893.6781006, -350.6282654, 1421.1749268, -1879.8244629, 2244.3063965
1: -559.9286499, 2114.3937988, -429.7668762, 1586.6865234, -2146.6152344, 2543.3227539
2: -650.8837891, 2142.2656250, -492.2691956, 1613.2059326, -2264.0893555, 2634.5349121
3: -919.2867432, 2355.4443359, -698.5472412, 1759.9062500, -2679.1928711, 3053.9916992
4: -1107.5815430, 2179.3068848, -827.3372803, 1641.0731201, -2748.6545410, 3006.6437988

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B2_A2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9501667, upper bound: 1757.0943939
time: 0.73 seconds

## Relational analysis of IS_B1_A2_B2_A2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9501667, upper bound: 1757.0960066
time: 0.68 seconds

## BFS IS instance: IS_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -447.9715271, 1847.6967773, -352.7528076, 1429.8717041, -1877.8432617, 2200.4497070
1: -547.8370361, 2063.1066895, -432.3590393, 1596.3819580, -2144.2189941, 2495.4653320
2: -635.2952881, 2091.4885254, -495.2670288, 1623.0344238, -2258.3295898, 2586.7553711
3: -897.7854614, 2294.9624023, -702.8384399, 1770.8116455, -2668.5969238, 2997.8005371
4: -1077.4190674, 2128.1311035, -832.5136719, 1651.0700684, -2728.4892578, 2960.6447754

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B2_A2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0377739, upper bound: 1757.1128230
time: 0.69 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0377588, upper bound: 1757.1143891
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -312.8979492, 1270.4250488, -394.0575562, 1624.7408447, -1937.6386719, 1664.4826660
1: -383.3760681, 1418.4169922, -481.0112000, 1813.6910400, -2197.0671387, 1899.4279785
2: -438.4498291, 1441.6322021, -559.5606079, 1838.4639893, -2276.9133301, 2001.1927490
3: -622.5888672, 1570.7271729, -788.6979370, 2019.8027344, -2642.3911133, 2359.4250488
4: -735.8646851, 1465.9256592, -950.7693481, 1869.5782471, -2605.4428711, 2416.6948242

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059715, upper bound: 1756.9420540
time: 0.66 seconds

## Relational analysis of IS_B2_A1_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055435, upper bound: 1756.9453307
time: 0.77 seconds

## BFS IS instance: IS_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -312.8979492, 1270.4250488, -458.6496277, 1893.6781006, -2206.5761719, 1729.0745850
1: -383.3760681, 1418.4169922, -559.9286499, 2114.3937988, -2496.8854980, 1978.3453369
2: -438.4498291, 1441.6322021, -650.8837891, 2142.2656250, -2580.7150879, 2092.5161133
3: -622.5888672, 1570.7271729, -919.2867432, 2355.4443359, -2978.0329590, 2490.0139160
4: -735.8646851, 1465.9256592, -1107.5815430, 2179.3068848, -2915.1716309, 2573.5070801

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059715, upper bound: 1757.0723120
time: 0.72 seconds

## Relational analysis of IS_B2_A1_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055435, upper bound: 1757.0803960
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -391.1911621, 1608.9113770, -1923.6638184, 1669.1890869
1: -385.6337891, 1426.8695068, -478.6294861, 1796.3077393, -2181.9414062, 1905.4986572
2: -441.0567932, 1450.1931152, -554.5004883, 1822.2730713, -2263.3298340, 2004.6936035
3: -626.3041992, 1580.1624756, -782.7835693, 1996.8884277, -2623.1926270, 2362.9460449
4: -740.3444824, 1474.6290283, -938.0429688, 1854.0861816, -2594.4306641, 2412.6711426

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1197587, upper bound: 1757.0370100
time: 0.70 seconds

## Relational analysis of IS_B2_A1_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198223, upper bound: 1757.0393753
time: 0.67 seconds

## BFS IS instance: IS_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -447.9715271, 1847.6967773, -2162.4489746, 1725.9696045
1: -385.6337891, 1426.8695068, -547.8370361, 2063.1066895, -2448.7402344, 1974.7064209
2: -441.0567932, 1450.1931152, -635.2952881, 2091.4885254, -2532.5454102, 2085.4882812
3: -626.3041992, 1580.1624756, -897.7854614, 2294.9624023, -2921.2663574, 2477.9475098
4: -740.3444824, 1474.6290283, -1077.4190674, 2128.1311035, -2868.4755859, 2552.0478516

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1197587, upper bound: 1757.1028594
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198222, upper bound: 1757.1063029
time: 0.67 seconds

## BFS IS instance: IS_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -350.6282654, 1421.1749268, -394.0575562, 1624.7408447, -1975.3691406, 1815.2324219
1: -429.7668762, 1586.6865234, -481.0112000, 1813.6910400, -2243.4580078, 2067.6977539
2: -492.2691956, 1613.2059326, -559.5606079, 1838.4639893, -2330.7331543, 2172.7663574
3: -698.5472412, 1759.9062500, -788.6979370, 2019.8027344, -2718.3500977, 2548.6042480
4: -827.3372803, 1641.0731201, -950.7693481, 1869.5782471, -2696.9155273, 2591.8422852

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A2_B1_B1_A1

### Relational analysis result of IS_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0851329, upper bound: 1756.9283175
time: 0.68 seconds

## Relational analysis of IS_B2_A1_A2_B1_B1_A2

### Relational analysis result of IS_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0870045, upper bound: 1756.9283175
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -391.1911621, 1608.9113770, -1961.6641846, 1821.0628662
1: -432.3590393, 1596.3819580, -478.6294861, 1796.3077393, -2228.6665039, 2075.0114746
2: -495.2670288, 1623.0344238, -554.5004883, 1822.2730713, -2317.5400391, 2177.5349121
3: -702.8384399, 1770.8116455, -782.7835693, 1996.8884277, -2699.7268066, 2553.5952148
4: -832.5136719, 1651.0700684, -938.0429688, 1854.0861816, -2686.5998535, 2589.1125488

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A2_B1_B2_A1

### Relational analysis result of IS_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1116861, upper bound: 1757.0262657
time: 0.74 seconds

## Relational analysis of IS_B2_A1_A2_B1_B2_A2

### Relational analysis result of IS_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1132522, upper bound: 1757.0262457
time: 0.77 seconds

## BFS IS instance: IS_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -350.6282654, 1421.1749268, -458.6496277, 1893.6781006, -2244.3063965, 1879.8244629
1: -429.7668762, 1586.6865234, -559.9286499, 2114.3937988, -2543.3227539, 2146.6152344
2: -492.2691956, 1613.2059326, -650.8837891, 2142.2656250, -2634.5349121, 2264.0895996
3: -698.5472412, 1759.9062500, -919.2867432, 2355.4443359, -3053.9916992, 2679.1928711
4: -827.3372803, 1641.0731201, -1107.5815430, 2179.3068848, -3006.6437988, 2748.6545410

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A2_B2_B1_A1

### Relational analysis result of IS_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0943777, upper bound: 1756.9501544
time: 0.65 seconds

## Relational analysis of IS_B2_A1_A2_B2_B1_A2

### Relational analysis result of IS_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0926970, upper bound: 1756.9471116
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -447.9715271, 1847.6967773, -2200.4497070, 1877.8432617
1: -432.3590393, 1596.3819580, -547.8370361, 2063.1066895, -2495.4653320, 2144.2189941
2: -495.2670288, 1623.0344238, -635.2952881, 2091.4885254, -2586.7553711, 2258.3295898
3: -702.8384399, 1770.8116455, -897.7854614, 2294.9624023, -2997.8005371, 2668.5969238
4: -832.5136719, 1651.0700684, -1077.4190674, 2128.1311035, -2960.6447754, 2728.4892578

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1127358, upper bound: 1757.0377788
time: 1.08 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2_A2

### Relational analysis result of IS_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1132522, upper bound: 1757.0364813
time: 0.71 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -391.1510315, 1608.5513916, -458.6496277, 1893.6781006, -2284.8291016, 2067.2004395
1: -478.6072083, 1795.9305420, -559.9286499, 2114.3937988, -2591.9038086, 2355.8588867
2: -554.4152222, 1821.9239502, -650.8837891, 2142.2656250, -2696.6809082, 2472.8076172
3: -782.6508179, 1996.3046875, -919.2867432, 2355.4443359, -3138.0952148, 2915.5913086
4: -937.7144775, 1853.8281250, -1107.5815430, 2179.3068848, -3117.0209961, 2961.4096680

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A1_B2_B1_B1

### Relational analysis result of IS_B2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9375802, upper bound: 1757.0728053
time: 0.66 seconds

## Relational analysis of IS_B2_A2_A1_B2_B1_B2

### Relational analysis result of IS_B2_A2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1756.9239628, upper bound: 1756.9410183
time: 0.75 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -393.3483887, 1617.7186279, -447.9715271, 1847.6967773, -2241.0449219, 2065.6901855
1: -481.2612000, 1806.1429443, -547.8370361, 2063.1066895, -2544.3676758, 2353.9799805
2: -557.5192261, 1832.2569580, -635.2952881, 2091.4885254, -2649.0075684, 2467.5522461
3: -787.0695801, 2007.7425537, -897.7854614, 2294.9624023, -3082.0319824, 2905.5273438
4: -943.0974121, 1864.2763672, -1077.4190674, 2128.1311035, -3071.2280273, 2941.6953125

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0253599, upper bound: 1757.0989573
time: 0.63 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0224325, upper bound: 1757.0328088
time: 0.93 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -458.6496277, 1893.6781006, -391.1510315, 1608.5513916, -2067.2004395, 2284.8291016
1: -559.9286499, 2114.3937988, -478.6072083, 1795.9305420, -2355.8588867, 2591.9038086
2: -650.8837891, 2142.2656250, -554.4152222, 1821.9239502, -2472.8076172, 2696.6809082
3: -919.2867432, 2355.4443359, -782.6508179, 1996.3046875, -2915.5913086, 3138.0952148
4: -1107.5815430, 2179.3068848, -937.7144775, 1853.8281250, -2961.4096680, 3117.0209961

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0758096, upper bound: 1757.0245712
time: 0.72 seconds

## Relational analysis of IS_B2_A2_A2_B1_A1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1756.9454973, upper bound: 1757.0186760
time: 0.84 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -447.9715271, 1847.6967773, -393.3483887, 1617.7186279, -2065.6901855, 2241.0446777
1: -547.8370361, 2063.1066895, -481.2612000, 1806.1429443, -2353.9799805, 2544.3676758
2: -635.2952881, 2091.4885254, -557.5192261, 1832.2569580, -2467.5522461, 2649.0075684
3: -897.7854614, 2294.9624023, -787.0695801, 2007.7425537, -2905.5273438, 3082.0319824
4: -1077.4190674, 2128.1311035, -943.0974121, 1864.2763672, -2941.6953125, 3071.2280273

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0993992, upper bound: 1757.0256050
time: 0.67 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0328087, upper bound: 1757.0224325
time: 0.70 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -458.6496277, 1893.6781006, -447.8392029, 1846.7592773, -2305.4082031, 2341.5173340
1: -559.9286499, 2114.3937988, -547.6820068, 2062.0759277, -2622.0043945, 2660.7995605
2: -650.8837891, 2142.2656250, -635.0977783, 2090.4782715, -2741.3620605, 2777.3632812
3: -919.2867432, 2355.4443359, -897.4064331, 2293.6945801, -3212.9812012, 3252.8508301
4: -1107.5815430, 2179.3068848, -1076.8824463, 2127.1098633, -3234.6914062, 3256.1892090

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1756.9467174, upper bound: 1757.0313450
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1756.9466342, upper bound: 1757.0301891
time: 0.66 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -447.9715271, 1847.6967773, -450.1930237, 1856.5698242, -2304.5412598, 2297.8896484
1: -547.8370361, 2063.1066895, -550.5508423, 2073.0170898, -2620.8540039, 2613.6572266
2: -635.2952881, 2091.4885254, -638.3911133, 2101.5507812, -2736.8461914, 2729.8796387
3: -897.7854614, 2294.9624023, -902.1766968, 2305.8959961, -3203.6813965, 3197.1389160
4: -1077.4190674, 2128.1311035, -1082.5723877, 2138.3459473, -3215.7651367, 3210.7031250

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.74 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.80 seconds
IS_B1_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1076380, upper bound: 1757.1216996
IS_B1_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1244855, upper bound: 1757.1221901
IS_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1226702, upper bound: 1757.1082130
IS_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1245527, upper bound: 1757.1245527
IS_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1145709, upper bound: 1757.1240235
IS_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1159619, upper bound: 1757.1240035
IS_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1169335, upper bound: 1757.1238128
IS_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1183245, upper bound: 1757.1237928
IS_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1240235, upper bound: 1757.1145709
IS_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1240035, upper bound: 1757.1159619
IS_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1238128, upper bound: 1757.1169335
IS_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1237928, upper bound: 1757.1183245
IS_B1_A1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0654839, upper bound: 1757.0654839
IS_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0654839, upper bound: 1757.1174983
IS_B1_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1171477, upper bound: 1757.1185641
IS_B1_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1183245, upper bound: 1757.1185441
IS_B1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9420540, upper bound: 1757.1059715
IS_B1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9453307, upper bound: 1757.1055435
IS_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9420540, upper bound: 1757.1217634
IS_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9453307, upper bound: 1757.1227017
IS_B1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0370100, upper bound: 1757.1197588
IS_B1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1198223
IS_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0370100, upper bound: 1757.1234787
IS_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1240331
IS_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9283175, upper bound: 1757.0851329
IS_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9283175, upper bound: 1757.0870045
IS_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0262657, upper bound: 1757.1116861
IS_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0262457, upper bound: 1757.1132523
IS_B1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9501667, upper bound: 1757.0943939
IS_B1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9501667, upper bound: 1757.0960066
IS_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0377739, upper bound: 1757.1128230
IS_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0377588, upper bound: 1757.1143891
IS_B2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1059715, upper bound: 1756.9420540
IS_B2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1055435, upper bound: 1756.9453307
IS_B2_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1059715, upper bound: 1757.0723120
IS_B2_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1055435, upper bound: 1757.0803960
IS_B2_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1197587, upper bound: 1757.0370100
IS_B2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1198223, upper bound: 1757.0393753
IS_B2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1197587, upper bound: 1757.1028594
IS_B2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1198222, upper bound: 1757.1063029
IS_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0851329, upper bound: 1756.9283175
IS_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0870045, upper bound: 1756.9283175
IS_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1116861, upper bound: 1757.0262657
IS_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1132522, upper bound: 1757.0262457
IS_B2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0943777, upper bound: 1756.9501544
IS_B2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0926970, upper bound: 1756.9471116
IS_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1127358, upper bound: 1757.0377788
IS_B2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.1132522, upper bound: 1757.0364813
IS_B2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9375802, upper bound: 1757.0728053
IS_B2_A2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9239628, upper bound: 1756.9410183
IS_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0253599, upper bound: 1757.0989573
IS_B2_A2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0224325, upper bound: 1757.0328088
IS_B2_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0758096, upper bound: 1757.0245712
IS_B2_A2_A2_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9454973, upper bound: 1757.0186760
IS_B2_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0993992, upper bound: 1757.0256050
IS_B2_A2_A2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0328087, upper bound: 1757.0224325
IS_B2_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9467174, upper bound: 1757.0313450
IS_B2_A2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 0, lower bound: -1756.9466342, upper bound: 1757.0301891
IS_B2_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
IS_B2_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456

## BFS IS instance: IS_B1_A1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -272.9577026, 1104.0373535, -309.7024231, 1256.9619141, -1529.9194336, 1413.7397461
1: -333.7005005, 1232.6545410, -379.4102173, 1403.3479004, -1737.0482178, 1612.0646973
2: -382.0137024, 1253.4129639, -434.0346069, 1426.3626709, -1808.3763428, 1687.4475098
3: -540.7861938, 1365.3823242, -616.1664429, 1554.3078613, -2095.0939941, 1981.5488281
4: -640.4468384, 1274.7979736, -728.6403809, 1450.4095459, -2090.8559570, 2003.4383545

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0843898, upper bound: 1757.1114380
time: 0.65 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1052754, upper bound: 1757.1216324
time: 0.65 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1052754, upper bound: 1757.1216996
time: 0.63 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -301.9649963, 1228.8576660, -313.3843689, 1272.5599365, -1574.5249023, 1542.2420654
1: -369.8095398, 1371.7869873, -383.9473267, 1420.7854004, -1790.5947266, 1755.7341309
2: -423.6985779, 1394.3264160, -439.1740417, 1444.0181885, -1867.7167969, 1833.5004883
3: -600.9874268, 1519.7431641, -623.5877075, 1573.4835205, -2174.4707031, 2143.3303223
4: -711.5792847, 1417.9205322, -737.2165527, 1468.3677979, -2179.9465332, 2155.1369629

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1229139, upper bound: 1757.1218367
time: 0.77 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244807, upper bound: 1757.1221348
time: 0.69 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -309.7541199, 1257.2274170, -284.8586121, 1149.5782471, -1459.3321533, 1542.0860596
1: -379.4916992, 1403.6440430, -348.4722595, 1283.7385254, -1663.2302246, 1752.1163330
2: -434.0444641, 1426.6933594, -398.1148987, 1305.2644043, -1739.3087158, 1824.8081055
3: -616.2919922, 1554.5042725, -564.3104248, 1421.2558594, -2037.5477295, 2118.8146973
4: -728.6088257, 1450.7320557, -666.8779297, 1327.5009766, -2056.1093750, 2117.6096191

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1224207, upper bound: 1757.1024053
time: 0.65 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1226465, upper bound: 1757.1020788
time: 0.69 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -313.4305115, 1272.7999268, -313.1314697, 1271.2943115, -1584.7248535, 1585.9313965
1: -384.0217285, 1421.0507812, -383.6576233, 1419.3879395, -1803.4096680, 1804.7081299
2: -439.1771240, 1444.3166504, -438.7846985, 1442.6059570, -1881.7830811, 1883.1011963
3: -623.7006836, 1573.6455078, -623.0579834, 1571.8681641, -2195.5683594, 2196.7033691
4: -737.1726074, 1468.6575928, -736.4831543, 1466.9417725, -2204.1140137, 2205.1406250

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181179, upper bound: 1757.1153950
time: 0.70 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1152449, upper bound: 1757.1152449
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -322.3203430, 1304.5732422, -302.9513245, 1232.9606934, -1555.2808838, 1607.5245361
1: -394.6984558, 1456.8837891, -371.0130005, 1376.3614502, -1771.0599365, 1827.8966064
2: -454.1062012, 1480.6278076, -425.0845947, 1398.9653320, -1853.0714111, 1905.7124023
3: -641.2573853, 1618.7316895, -602.9675903, 1524.8070068, -2166.0639648, 2221.6992188
4: -762.6246338, 1508.0877686, -713.9273682, 1422.6154785, -2185.2402344, 2222.0151367

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112782, upper bound: 1757.1235873
time: 0.71 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1140804, upper bound: 1757.1043430
time: 0.70 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1145709, upper bound: 1757.1240235
time: 0.78 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -341.7520752, 1385.8635254, -303.3241882, 1234.4622803, -1576.2143555, 1689.1877441
1: -418.9904785, 1547.2274170, -371.4692993, 1378.0418701, -1797.0319824, 1918.6965332
2: -479.8916321, 1573.2951660, -425.6004028, 1400.6689453, -1880.5605469, 1998.8955078
3: -681.0548706, 1716.4506836, -603.7112427, 1526.6672363, -2207.7221680, 2320.1613770
4: -806.5220947, 1600.7314453, -714.8023071, 1424.3453369, -2230.8671875, 2315.5336914

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1154714, upper bound: 1757.1043230
time: 0.75 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1159619, upper bound: 1757.1240035
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -323.3041992, 1308.3963623, -313.0219116, 1271.1387939, -1594.4428711, 1621.4182129
1: -395.9123230, 1461.1677246, -383.5216675, 1419.1925049, -1815.1046143, 1844.6892090
2: -455.4548340, 1484.9721680, -438.6071472, 1442.4321289, -1897.8868408, 1923.5791016
3: -643.2018433, 1623.4108887, -622.8839111, 1571.5855713, -2214.7873535, 2246.2946777
4: -764.8485718, 1512.4916992, -736.2062378, 1466.7429199, -2231.5915527, 2248.6977539

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1139749, upper bound: 1757.1233894
time: 0.95 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1150510, upper bound: 1757.1041889
time: 1.10 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1169335, upper bound: 1757.1238128
time: 0.76 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -342.8786621, 1390.2818604, -313.4305115, 1272.7999268, -1615.6783447, 1703.7124023
1: -420.3764038, 1552.1739502, -384.0217285, 1421.0507812, -1841.4270020, 1936.1956787
2: -481.4416199, 1578.3128662, -439.1771240, 1444.3166504, -1925.7583008, 2017.4899902
3: -683.2790527, 1721.8658447, -623.7006836, 1573.6455078, -2256.9243164, 2345.5664062
4: -809.0892334, 1605.8089600, -737.1726074, 1468.6575928, -2277.7468262, 2342.9814453

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1164421, upper bound: 1757.1041688
time: 0.69 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1183245, upper bound: 1757.1237928
time: 0.71 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -302.9513245, 1232.9606934, -322.3203430, 1304.5732422, -1607.5245361, 1555.2808838
1: -371.0130005, 1376.3614502, -394.6984558, 1456.8837891, -1827.8966064, 1771.0599365
2: -425.0845947, 1398.9653320, -454.1062012, 1480.6278076, -1905.7124023, 1853.0714111
3: -602.9675903, 1524.8070068, -641.2573853, 1618.7316895, -2221.6992188, 2166.0639648
4: -713.9273682, 1422.6154785, -762.6246338, 1508.0877686, -2222.0151367, 2185.2402344

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235873, upper bound: 1757.1112782
time: 0.71 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1043430, upper bound: 1757.1140804
time: 0.91 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1240235, upper bound: 1757.1145709
time: 0.85 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -303.3241882, 1234.4622803, -341.7520752, 1385.8635254, -1689.1876221, 1576.2143555
1: -371.4692993, 1378.0418701, -418.9904785, 1547.2274170, -1918.6965332, 1797.0319824
2: -425.6004028, 1400.6689453, -479.8916321, 1573.2951660, -1998.8955078, 1880.5605469
3: -603.7112427, 1526.6672363, -681.0548706, 1716.4506836, -2320.1613770, 2207.7221680
4: -714.8023071, 1424.3453369, -806.5220947, 1600.7314453, -2315.5336914, 2230.8671875

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1043230, upper bound: 1757.1154714
time: 0.69 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1240035, upper bound: 1757.1159619
time: 0.75 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -313.0219116, 1271.1387939, -323.3041992, 1308.3963623, -1621.4182129, 1594.4428711
1: -383.5216675, 1419.1925049, -395.9123230, 1461.1677246, -1844.6892090, 1815.1046143
2: -438.6071472, 1442.4321289, -455.4548340, 1484.9721680, -1923.5791016, 1897.8868408
3: -622.8839111, 1571.5855713, -643.2018433, 1623.4108887, -2246.2946777, 2214.7873535
4: -736.2062378, 1466.7429199, -764.8485718, 1512.4916992, -2248.6977539, 2231.5915527

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233894, upper bound: 1757.1139749
time: 0.71 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1041889, upper bound: 1757.1150510
time: 0.67 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1238128, upper bound: 1757.1169335
time: 0.80 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -313.4305115, 1272.7999268, -342.8786621, 1390.2818604, -1703.7124023, 1615.6784668
1: -384.0217285, 1421.0507812, -420.3764038, 1552.1739502, -1936.1956787, 1841.4270020
2: -439.1771240, 1444.3166504, -481.4416199, 1578.3128662, -2017.4899902, 1925.7583008
3: -623.7006836, 1573.6455078, -683.2790527, 1721.8658447, -2345.5664062, 2256.9245605
4: -737.1726074, 1468.6575928, -809.0892334, 1605.8089600, -2342.9814453, 2277.7468262

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1041688, upper bound: 1757.1164421
time: 0.72 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237928, upper bound: 1757.1183245
time: 0.67 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -342.4194946, 1390.7973633, -350.8810730, 1422.3142090, -1764.7336426, 1741.6784668
1: -419.5917969, 1552.5744629, -430.0623474, 1587.9256592, -2007.5172119, 1982.6368408
2: -481.1055603, 1578.5659180, -492.6302185, 1614.4853516, -2095.5908203, 2071.1960449
3: -682.4114380, 1722.1546631, -699.1375732, 1761.4387207, -2443.8496094, 2421.2917480
4: -808.9281616, 1605.6789551, -828.0973511, 1642.4067383, -2451.3344727, 2433.7763672

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0640203, upper bound: 1757.0700951
time: 0.63 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0640198, upper bound: 1757.1173627
time: 0.75 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -321.1210938, 1299.7056885, -352.3422546, 1428.2148438, -1749.3359375, 1652.0479736
1: -393.2482605, 1451.4301758, -431.8547974, 1594.5312500, -1987.7792969, 1883.2849121
2: -452.3868408, 1475.1323242, -494.6956482, 1621.1538086, -2073.5405273, 1969.8280029
3: -638.9245605, 1612.5576172, -702.0164795, 1768.7337646, -2407.6582031, 2314.5739746
4: -759.7264404, 1502.4771729, -831.5426025, 1649.1555176, -2408.8818359, 2334.0195312

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A2_A2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1141784, upper bound: 1757.1181407
time: 0.71 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A2_A2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1170175, upper bound: 1757.1171531
time: 0.81 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1170175, upper bound: 1757.1185441
time: 0.82 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -341.0588989, 1383.0490723, -352.7528076, 1429.8717041, -1770.9302979, 1735.8018799
1: -418.1606445, 1544.0767822, -432.3590393, 1596.3819580, -2014.5426025, 1976.4356689
2: -478.9018860, 1570.1264648, -495.2670288, 1623.0344238, -2101.9362793, 2065.3933105
3: -679.7196655, 1712.8656006, -702.8384399, 1770.8116455, -2450.5312500, 2415.7041016
4: -804.8374023, 1597.4860840, -832.5136719, 1651.0700684, -2455.9074707, 2429.9997559

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1170175, upper bound: 1757.1171531
time: 0.72 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1170175, upper bound: 1757.1185441
time: 0.68 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -393.2033386, 1621.6105957, -301.5788879, 1227.2983398, -1620.5017090, 1923.1894531
1: -479.9403992, 1810.1788330, -369.3363342, 1370.0445557, -1849.9849854, 2179.5148926
2: -558.4140625, 1834.8607178, -423.1473694, 1392.5787354, -1950.9927979, 2258.0073242
3: -787.0300293, 2015.9833984, -600.1939697, 1517.7532959, -2304.7832031, 2616.1772461
4: -948.9454956, 1865.9179688, -710.5774536, 1416.1265869, -2365.0720215, 2576.4953613

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9414481, upper bound: 1757.1058256
time: 0.66 seconds

## Relational analysis of IS_B1_A2_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9420540, upper bound: 1757.1059715
time: 1.01 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -394.0575562, 1624.7408447, -311.5715637, 1265.2158203, -1659.2734375, 1936.3122559
1: -481.0112000, 1813.6910400, -381.7588501, 1412.5870361, -1893.5981445, 2195.4499512
2: -559.5606079, 1838.4639893, -436.5648499, 1435.7409668, -1995.3015137, 2275.0283203
3: -788.6979370, 2019.8027344, -619.9768677, 1564.1958008, -2352.8937988, 2639.7795410
4: -950.7693481, 1869.5782471, -732.6840210, 1459.9373779, -2410.7067871, 2602.2622070

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9451612, upper bound: 1757.1055435
time: 0.76 seconds

## Relational analysis of IS_B1_A2_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A2_B1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1756.9409235, upper bound: 1757.0302693
time: 0.73 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -457.8761292, 1890.7889404, -301.5788879, 1227.2983398, -1685.1744385, 2192.3679199
1: -558.9644775, 2111.1508789, -369.3363342, 1370.0445557, -1929.0086670, 2479.4836426
2: -649.8258057, 2138.9538574, -423.1473694, 1392.5787354, -2042.4045410, 2562.1013184
3: -917.7660522, 2351.8950195, -600.1939697, 1517.7532959, -2435.5190430, 2952.0888672
4: -1105.8704834, 2175.9438477, -710.5774536, 1416.1265869, -2521.9968262, 2886.5212402

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0705185, upper bound: 1757.1207893
time: 0.64 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9471400, upper bound: 1757.1103760
time: 0.64 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -458.6496277, 1893.6781006, -311.5715637, 1265.2158203, -1723.8653564, 2205.2497559
1: -559.9286499, 2114.3937988, -381.7588501, 1412.5870361, -1972.5153809, 2495.2729492
2: -650.8837891, 2142.2656250, -436.5648499, 1435.7409668, -2086.6247559, 2578.8300781
3: -919.2867432, 2355.4443359, -619.9768677, 1564.1958008, -2483.4824219, 2975.4211426
4: -1107.5815430, 2179.3068848, -732.6840210, 1459.9373779, -2567.5190430, 2911.9909668

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0790023, upper bound: 1757.1208096
time: 0.70 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9500836, upper bound: 1757.1085349
time: 0.70 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -390.2587891, 1605.3507080, -303.3241882, 1234.4622803, -1624.7210693, 1908.6749268
1: -477.4616089, 1792.3194580, -371.4692993, 1378.0418701, -1855.5032959, 2163.7878418
2: -553.2289429, 1818.2067871, -425.6004028, 1400.6689453, -1953.8979492, 2243.8071289
3: -780.9440308, 1992.5296631, -603.7112427, 1526.6672363, -2307.6113281, 2596.2404785
4: -935.9873657, 1849.9339600, -714.8023071, 1424.3453369, -2360.3327637, 2564.7363281

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0369009, upper bound: 1757.1119036
time: 0.79 seconds

## Relational analysis of IS_B1_A2_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0364006, upper bound: 1757.1102257
time: 0.65 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -391.1911621, 1608.9113770, -313.4305115, 1272.7999268, -1663.9909668, 1922.3419189
1: -478.6294861, 1796.3077393, -384.0217285, 1421.0507812, -1899.6799316, 2180.3291016
2: -554.5004883, 1822.2730713, -439.1771240, 1444.3166504, -1998.8171387, 2261.4497070
3: -782.7835693, 1996.8884277, -623.7006836, 1573.6455078, -2356.4289551, 2620.5891113
4: -938.0429688, 1854.0861816, -737.1726074, 1468.6575928, -2406.6997070, 2591.2587891

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0393753, upper bound: 1757.1131148
time: 0.96 seconds

## Relational analysis of IS_B1_A2_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0392252, upper bound: 1757.1102257
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -447.1631165, 1844.6667480, -303.3241882, 1234.4622803, -1681.6252441, 2147.9909668
1: -546.8303833, 2059.7094727, -371.4692993, 1378.0418701, -1924.8720703, 2431.1784668
2: -634.1912842, 2088.0224609, -425.6004028, 1400.6689453, -2034.8602295, 2513.6228027
3: -896.2004395, 2291.2534180, -603.7112427, 1526.6672363, -2422.8676758, 2894.9643555
4: -1075.6295166, 2124.6088867, -714.8023071, 1424.3453369, -2499.9748535, 2839.4111328

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0998943, upper bound: 1757.1226834
time: 0.66 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0351766, upper bound: 1757.1201904
time: 0.75 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -447.9715271, 1847.6967773, -313.4305115, 1272.7999268, -1720.7714844, 2161.1269531
1: -547.8370361, 2063.1066895, -384.0217285, 1421.0507812, -1968.8876953, 2447.1279297
2: -635.2952881, 2091.4885254, -439.1771240, 1444.3166504, -2079.6118164, 2530.6652832
3: -897.7854614, 2294.9624023, -623.7006836, 1573.6455078, -2471.4304199, 2918.6628418
4: -1077.4190674, 2128.1311035, -737.1726074, 1468.6575928, -2546.0764160, 2865.3037109

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1041296, upper bound: 1757.1226441
time: 0.70 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0375392, upper bound: 1757.1199797
time: 0.73 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -393.7021790, 1623.2320557, -321.0876465, 1299.3459473, -1693.0480957, 1944.3197021
1: -480.5818176, 1812.0107422, -393.2141724, 1451.0750732, -1931.6568604, 2205.2248535
2: -559.0551758, 1836.7639160, -452.3346863, 1474.7414551, -2033.7966309, 2289.0986328
3: -787.9793091, 2017.9230957, -638.7474365, 1612.0501709, -2400.0295410, 2656.6704102
4: -949.8880005, 1867.8560791, -759.4749146, 1502.0783691, -2451.9663086, 2627.3310547

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9246956, upper bound: 1757.0804236
time: 0.69 seconds

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9281319, upper bound: 1757.0851329
time: 0.64 seconds

## Relational analysis of IS_B1_A2_B2_A1_A1_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9279698, upper bound: 1757.0846077
time: 0.68 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -394.0575562, 1624.7408447, -340.7836914, 1381.6943359, -1775.7519531, 1965.5242920
1: -481.0112000, 1813.6910400, -417.8229980, 1542.6003418, -2023.6113281, 2231.5141602
2: -559.5606079, 1838.4639893, -478.4844055, 1568.6036377, -2128.1643066, 2316.9479980
3: -788.6979370, 2019.8027344, -679.0448608, 1711.0463867, -2499.7441406, 2698.8471680
4: -950.7693481, 1869.5782471, -803.9799194, 1595.8851318, -2546.6545410, 2673.5581055

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9281319, upper bound: 1757.0870045
time: 0.67 seconds

## Relational analysis of IS_B1_A2_B2_A1_A1_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1756.9263171, upper bound: 1757.0556861
time: 0.63 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -390.8579407, 1607.5142822, -323.3041992, 1308.3963623, -1699.2542725, 1930.8183594
1: -478.2260742, 1794.7524414, -395.9123230, 1461.1677246, -1939.3937988, 2190.6647949
2: -554.0273438, 1820.6961670, -455.4548340, 1484.9721680, -2038.9995117, 2276.1506348
3: -782.1118774, 1995.1470947, -643.2018433, 1623.4108887, -2405.5227051, 2638.3483887
4: -937.2216187, 1852.4973145, -764.8485718, 1512.4916992, -2449.7131348, 2617.3459473

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0258423, upper bound: 1757.1084737
time: 0.71 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0262657, upper bound: 1757.1032503
time: 0.87 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0259539, upper bound: 1757.0932917
time: 0.63 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -391.1911621, 1608.9113770, -342.8786621, 1390.2818604, -1781.4729004, 1951.7897949
1: -478.6294861, 1796.3077393, -420.3764038, 1552.1739502, -2030.8031006, 2216.6838379
2: -554.5004883, 1822.2730713, -481.4416199, 1578.3128662, -2132.8134766, 2303.7145996
3: -782.7835693, 1996.8884277, -683.2790527, 1721.8658447, -2504.6494141, 2680.1674805
4: -938.0429688, 1854.0861816, -809.0892334, 1605.8089600, -2543.8513184, 2663.1752930

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0262457, upper bound: 1757.1042268
time: 2.37 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0259539, upper bound: 1757.0954868
time: 0.68 seconds

## BFS IS instance: IS_B1_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -458.1756897, 1891.7541504, -321.0876465, 1299.3459473, -1757.5216064, 2212.8415527
1: -559.3464966, 2112.2468262, -393.2141724, 1451.0750732, -2010.4216309, 2504.6352539
2: -650.2313232, 2140.0793457, -452.3346863, 1474.7414551, -2124.9724121, 2592.4140625
3: -918.3380737, 2353.0673828, -638.7474365, 1612.0501709, -2530.3881836, 2991.8139648
4: -1106.4755859, 2177.0839844, -759.4749146, 1502.0783691, -2608.5537109, 2936.5588379

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B2_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9501667, upper bound: 1757.0943939
time: 0.62 seconds

## Relational analysis of IS_B1_A2_B2_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9501667, upper bound: 1757.0943939
time: 0.66 seconds

## BFS IS instance: IS_B1_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -458.6496277, 1893.6781006, -340.7836914, 1381.6943359, -1840.3438721, 2234.4619141
1: -559.9286499, 2114.3937988, -417.8229980, 1542.6003418, -2102.5290527, 2531.3947754
2: -650.8837891, 2142.2656250, -478.4844055, 1568.6036377, -2219.4873047, 2620.7497559
3: -919.2867432, 2355.4443359, -679.0448608, 1711.0463867, -2630.3325195, 3034.4890137
4: -1107.5815430, 2179.3068848, -803.9799194, 1595.8851318, -2703.4667969, 2983.2866211

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B2_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9501667, upper bound: 1757.0960066
time: 0.72 seconds

## Relational analysis of IS_B1_A2_B2_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9501667, upper bound: 1757.0960066
time: 0.73 seconds

## BFS IS instance: IS_B1_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -447.5048523, 1845.7918701, -323.3041992, 1308.3963623, -1755.9011230, 2169.0961914
1: -547.2626343, 2060.9782715, -395.9123230, 1461.1677246, -2008.4299316, 2456.8906250
2: -634.6489868, 2089.3256836, -455.4548340, 1484.9721680, -2119.6210938, 2544.7805176
3: -896.8460083, 2292.6035156, -643.2018433, 1623.4108887, -2520.2568359, 2935.8051758
4: -1076.3200684, 2125.9289551, -764.8485718, 1512.4916992, -2588.8112793, 2890.7775879

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B2_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0377588, upper bound: 1757.1128230
time: 1.03 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0377588, upper bound: 1757.1128230
time: 0.77 seconds

## BFS IS instance: IS_B1_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -447.9715271, 1847.6967773, -342.8786621, 1390.2818604, -1838.2534180, 2190.5754395
1: -547.8370361, 2063.1066895, -420.3764038, 1552.1739502, -2100.0104980, 2483.4824219
2: -635.2952881, 2091.4885254, -481.4416199, 1578.3128662, -2213.6081543, 2572.9301758
3: -897.7854614, 2294.9624023, -683.2790527, 1721.8658447, -2619.6513672, 2978.2412109
4: -1077.4190674, 2128.1311035, -809.0892334, 1605.8089600, -2683.2280273, 2937.2202148

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_B2_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0377588, upper bound: 1757.1143891
time: 0.71 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0377588, upper bound: 1757.1143891
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -301.5788879, 1227.2983398, -393.2033386, 1621.6105957, -1923.1894531, 1620.5017090
1: -369.3363342, 1370.0445557, -479.9403992, 1810.1788330, -2179.5148926, 1849.9849854
2: -423.1473694, 1392.5787354, -558.4140625, 1834.8607178, -2258.0075684, 1950.9927979
3: -600.1939697, 1517.7532959, -787.0300293, 2015.9833984, -2616.1772461, 2304.7832031
4: -710.5774536, 1416.1265869, -948.9454956, 1865.9179688, -2576.4953613, 2365.0720215

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A1_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058256, upper bound: 1756.9414481
time: 0.74 seconds

## Relational analysis of IS_B2_A1_A1_B1_B1_A1_A2

### Relational analysis result of IS_B2_A1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059715, upper bound: 1756.9420540
time: 0.73 seconds

## BFS IS instance: IS_B2_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -311.5715637, 1265.2158203, -394.0575562, 1624.7408447, -1936.3121338, 1659.2734375
1: -381.7588501, 1412.5870361, -481.0112000, 1813.6910400, -2195.4497070, 1893.5981445
2: -436.5648499, 1435.7409668, -559.5606079, 1838.4639893, -2275.0283203, 1995.3015137
3: -619.9768677, 1564.1958008, -788.6979370, 2019.8027344, -2639.7795410, 2352.8937988
4: -732.6840210, 1459.9373779, -950.7693481, 1869.5782471, -2602.2622070, 2410.7067871

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055435, upper bound: 1756.9451612
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A1_B1_B1_A2_A2

### Relational analysis result of IS_B2_A1_A1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0302693, upper bound: 1756.9409235
time: 0.77 seconds

## BFS IS instance: IS_B2_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -301.5788879, 1227.2983398, -457.8761292, 1890.7889404, -2192.3679199, 1685.1744385
1: -369.3363342, 1370.0445557, -558.9644775, 2111.1508789, -2479.4836426, 1929.0086670
2: -423.1473694, 1392.5787354, -649.8258057, 2138.9538574, -2562.1010742, 2042.4045410
3: -600.1939697, 1517.7532959, -917.7660522, 2351.8950195, -2952.0888672, 2435.5190430
4: -710.5774536, 1416.1265869, -1105.8704834, 2175.9438477, -2886.5212402, 2521.9968262

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B1_B2_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1207893, upper bound: 1757.0705185
time: 0.70 seconds

## Relational analysis of IS_B2_A1_A1_B1_B2_A1_B2

### Relational analysis result of IS_B2_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1103761, upper bound: 1756.9471400
time: 0.72 seconds

## BFS IS instance: IS_B2_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -311.5715637, 1265.2158203, -458.6496277, 1893.6781006, -2205.2497559, 1723.8653564
1: -381.7588501, 1412.5870361, -559.9286499, 2114.3937988, -2495.2729492, 1972.5152588
2: -436.5648499, 1435.7409668, -650.8837891, 2142.2656250, -2578.8300781, 2086.6247559
3: -619.9768677, 1564.1958008, -919.2867432, 2355.4443359, -2975.4211426, 2483.4824219
4: -732.6840210, 1459.9373779, -1107.5815430, 2179.3068848, -2911.9909668, 2567.5190430

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B1_B2_A2_B1

### Relational analysis result of IS_B2_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1208096, upper bound: 1757.0790023
time: 0.78 seconds

## Relational analysis of IS_B2_A1_A1_B1_B2_A2_B2

### Relational analysis result of IS_B2_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1085350, upper bound: 1756.9500836
time: 0.65 seconds

## BFS IS instance: IS_B2_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -303.3241882, 1234.4622803, -390.2587891, 1605.3507080, -1908.6749268, 1624.7210693
1: -371.4692993, 1378.0418701, -477.4616089, 1792.3194580, -2163.7880859, 1855.5032959
2: -425.6004028, 1400.6689453, -553.2289429, 1818.2067871, -2243.8071289, 1953.8979492
3: -603.7112427, 1526.6672363, -780.9440308, 1992.5296631, -2596.2404785, 2307.6113281
4: -714.8023071, 1424.3453369, -935.9873657, 1849.9339600, -2564.7363281, 2360.3327637

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A1_B2_B1_A1_A1

### Relational analysis result of IS_B2_A1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1119037, upper bound: 1757.0369009
time: 0.74 seconds

## Relational analysis of IS_B2_A1_A1_B2_B1_A1_A2

### Relational analysis result of IS_B2_A1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1102257, upper bound: 1757.0364006
time: 0.71 seconds

## BFS IS instance: IS_B2_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -313.4305115, 1272.7999268, -391.1911621, 1608.9113770, -1922.3419189, 1663.9909668
1: -384.0217285, 1421.0507812, -478.6294861, 1796.3077393, -2180.3291016, 1899.6799316
2: -439.1771240, 1444.3166504, -554.5004883, 1822.2730713, -2261.4497070, 1998.8171387
3: -623.7006836, 1573.6455078, -782.7835693, 1996.8884277, -2620.5891113, 2356.4289551
4: -737.1726074, 1468.6575928, -938.0429688, 1854.0861816, -2591.2587891, 2406.6997070

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A1_B2_B1_A2_A1

### Relational analysis result of IS_B2_A1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1131148, upper bound: 1757.0393753
time: 0.68 seconds

## Relational analysis of IS_B2_A1_A1_B2_B1_A2_A2

### Relational analysis result of IS_B2_A1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1102257, upper bound: 1757.0392252
time: 0.70 seconds

## BFS IS instance: IS_B2_A1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -303.3241882, 1234.4622803, -447.1631165, 1844.6667480, -2147.9907227, 1681.6252441
1: -371.4692993, 1378.0418701, -546.8303833, 2059.7094727, -2431.1784668, 1924.8720703
2: -425.6004028, 1400.6689453, -634.1912842, 2088.0224609, -2513.6228027, 2034.8601074
3: -603.7112427, 1526.6672363, -896.2004395, 2291.2534180, -2894.9643555, 2422.8676758
4: -714.8023071, 1424.3453369, -1075.6295166, 2124.6088867, -2839.4111328, 2499.9748535

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B2_B2_A1_B1

### Relational analysis result of IS_B2_A1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1226834, upper bound: 1757.0998943
time: 0.62 seconds

## Relational analysis of IS_B2_A1_A1_B2_B2_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1201904, upper bound: 1757.0351766
time: 0.67 seconds

## BFS IS instance: IS_B2_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -313.4305115, 1272.7999268, -447.9715271, 1847.6967773, -2161.1271973, 1720.7714844
1: -384.0217285, 1421.0507812, -547.8370361, 2063.1066895, -2447.1279297, 1968.8878174
2: -439.1771240, 1444.3166504, -635.2952881, 2091.4885254, -2530.6652832, 2079.6118164
3: -623.7006836, 1573.6455078, -897.7854614, 2294.9624023, -2918.6628418, 2471.4306641
4: -737.1726074, 1468.6575928, -1077.4190674, 2128.1311035, -2865.3037109, 2546.0764160

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A1_B2_B2_A2_B1

### Relational analysis result of IS_B2_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1226441, upper bound: 1757.1041296
time: 0.71 seconds

## Relational analysis of IS_B2_A1_A1_B2_B2_A2_B2

### Relational analysis result of IS_B2_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1199797, upper bound: 1757.0375392
time: 0.70 seconds

## BFS IS instance: IS_B2_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -321.0876465, 1299.3459473, -393.7021790, 1623.2320557, -1944.3197021, 1693.0480957
1: -393.2141724, 1451.0750732, -480.5818176, 1812.0107422, -2205.2248535, 1931.6568604
2: -452.3346863, 1474.7414551, -559.0551758, 1836.7639160, -2289.0986328, 2033.7966309
3: -638.7474365, 1612.0501709, -787.9793091, 2017.9230957, -2656.6704102, 2400.0295410
4: -759.4749146, 1502.0783691, -949.8880005, 1867.8560791, -2627.3310547, 2451.9663086

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_B2_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0804236, upper bound: 1756.9246956
time: 0.62 seconds

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0851329, upper bound: 1756.9281319
time: 0.70 seconds

## Relational analysis of IS_B2_A1_A2_B1_B1_A1_A2

### Relational analysis result of IS_B2_A1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0846077, upper bound: 1756.9279698
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -340.7836914, 1381.6943359, -394.0575562, 1624.7408447, -1965.5244141, 1775.7519531
1: -417.8229980, 1542.6003418, -481.0112000, 1813.6910400, -2231.5141602, 2023.6113281
2: -478.4844055, 1568.6036377, -559.5606079, 1838.4639893, -2316.9479980, 2128.1643066
3: -679.0448608, 1711.0463867, -788.6979370, 2019.8027344, -2698.8471680, 2499.7441406
4: -803.9799194, 1595.8851318, -950.7693481, 1869.5782471, -2673.5581055, 2546.6545410

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0870045, upper bound: 1756.9281319
time: 0.78 seconds

## Relational analysis of IS_B2_A1_A2_B1_B1_A2_A2

### Relational analysis result of IS_B2_A1_A2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0556861, upper bound: 1756.9263171
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -323.3041992, 1308.3963623, -390.8579407, 1607.5142822, -1930.8183594, 1699.2542725
1: -395.9123230, 1461.1677246, -478.2260742, 1794.7524414, -2190.6647949, 1939.3936768
2: -455.4548340, 1484.9721680, -554.0273438, 1820.6961670, -2276.1506348, 2038.9993896
3: -643.2018433, 1623.4108887, -782.1118774, 1995.1470947, -2638.3483887, 2405.5227051
4: -764.8485718, 1512.4916992, -937.2216187, 1852.4973145, -2617.3459473, 2449.7131348

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_B2_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1084737, upper bound: 1757.0258423
time: 0.76 seconds

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1032504, upper bound: 1757.0262657
time: 0.71 seconds

## Relational analysis of IS_B2_A1_A2_B1_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0932916, upper bound: 1757.0259539
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -342.8786621, 1390.2818604, -391.1911621, 1608.9113770, -1951.7900391, 1781.4729004
1: -420.3764038, 1552.1739502, -478.6294861, 1796.3077393, -2216.6838379, 2030.8032227
2: -481.4416199, 1578.3128662, -554.5004883, 1822.2730713, -2303.7145996, 2132.8134766
3: -683.2790527, 1721.8658447, -782.7835693, 1996.8884277, -2680.1674805, 2504.6494141
4: -809.0892334, 1605.8089600, -938.0429688, 1854.0861816, -2663.1752930, 2543.8513184

Time for backsubstitution: 1.41 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407058]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1263229, upper bound: 1757.1264512
time: 0.76 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1264342, upper bound: 1757.1264342
time: 0.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.55 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 0, lower bound: -1757.1263229, upper bound: 1757.1264512
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 0, lower bound: -1757.1264342, upper bound: 1757.1264342

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -385.7852478, 1565.9726562, -352.9316101, 1436.0976562, -1821.8828125, 1918.9041748
1: -472.8652954, 1749.0377197, -432.6250916, 1604.0590820, -2076.9243164, 2181.6625977
2: -540.8838501, 1778.0942383, -494.3493042, 1629.8876953, -2170.7714844, 2272.4436035
3: -767.9857178, 1936.7133789, -702.1618652, 1774.0732422, -2542.0590820, 2638.8745117
4: -908.0595703, 1808.1075439, -828.9537354, 1656.8021240, -2564.8610840, 2637.0612793

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1263229, upper bound: 1757.1263229
time: 0.65 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1263229, upper bound: 1757.1264342
time: 0.67 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -394.7721863, 1602.8308105, -392.9946289, 1595.7901611, -1990.5621338, 1995.8254395
1: -483.8654175, 1790.2031250, -481.7033691, 1782.3461914, -2266.2114258, 2271.9064941
2: -553.5966187, 1819.9356689, -551.1360474, 1811.9309082, -2365.5270996, 2371.0715332
3: -786.0653076, 1982.4147949, -782.5770874, 1973.6967773, -2759.7622070, 2764.9919434
4: -929.4199829, 1850.5749512, -925.2904663, 1842.4506836, -2771.8706055, 2775.8654785

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1264342, upper bound: 1757.1263229
time: 0.76 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1264342, upper bound: 1757.1264342
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.82 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 0, lower bound: -1757.1263229, upper bound: 1757.1263229
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 0, lower bound: -1757.1263229, upper bound: 1757.1264342
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 0, lower bound: -1757.1264342, upper bound: 1757.1263229
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.82
Output dim: 0, lower bound: -1757.1264342, upper bound: 1757.1264342

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -352.9316101, 1436.0976562, -352.9316101, 1436.0976562, -1789.0291748, 1789.0291748
1: -432.6250916, 1604.0590820, -432.6250916, 1604.0590820, -2036.6842041, 2036.6842041
2: -494.3493042, 1629.8876953, -494.3493042, 1629.8876953, -2124.2368164, 2124.2368164
3: -702.1618652, 1774.0732422, -702.1618652, 1774.0732422, -2476.2351074, 2476.2351074
4: -828.9537354, 1656.8021240, -828.9537354, 1656.8021240, -2485.7553711, 2485.7553711

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0402457, upper bound: 1757.1079701
time: 0.70 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0360244
time: 0.79 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -392.9946289, 1595.7901611, -352.9316101, 1436.0976562, -1829.0922852, 1948.7218018
1: -481.7033691, 1782.3461914, -432.6250916, 1604.0590820, -2085.7622070, 2214.9709473
2: -551.1360474, 1811.9309082, -494.3493042, 1629.8876953, -2181.0236816, 2306.2795410
3: -782.5770874, 1973.6967773, -702.1618652, 1774.0732422, -2556.6503906, 2675.8586426
4: -925.2904663, 1842.4506836, -828.9537354, 1656.8021240, -2582.0925293, 2671.4042969

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1026017, upper bound: 1757.0402891
time: 0.62 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0398435
time: 0.62 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -352.9316101, 1436.0976562, -392.9946289, 1595.7901611, -1948.7218018, 1829.0922852
1: -432.6250916, 1604.0590820, -481.7033691, 1782.3461914, -2214.9707031, 2085.7624512
2: -494.3493042, 1629.8876953, -551.1360474, 1811.9309082, -2306.2792969, 2181.0236816
3: -702.1618652, 1774.0732422, -782.5770874, 1973.6967773, -2675.8586426, 2556.6503906
4: -828.9537354, 1656.8021240, -925.2904663, 1842.4506836, -2671.4042969, 2582.0925293

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0402457, upper bound: 1757.1026017
time: 0.79 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0889633
time: 0.67 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -392.9946289, 1595.7901611, -392.9946289, 1595.7901611, -1988.7847900, 1988.7847900
1: -481.7033691, 1782.3461914, -481.7033691, 1782.3461914, -2264.0493164, 2264.0495605
2: -551.1360474, 1811.9309082, -551.1360474, 1811.9309082, -2363.0666504, 2363.0666504
3: -782.5770874, 1973.6967773, -782.5770874, 1973.6967773, -2756.2739258, 2756.2739258
4: -925.2904663, 1842.4506836, -925.2904663, 1842.4506836, -2767.7412109, 2767.7412109

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0402457, upper bound: 1757.1219706
time: 0.66 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.1054668
time: 0.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.70 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -1757.0402457, upper bound: 1757.1079701
IS_B1_A1_B2, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0360244
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -1757.1026017, upper bound: 1757.0402891
IS_B1_A2_A2, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0398435
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -1757.0402457, upper bound: 1757.1026017
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0889633
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -1757.0402457, upper bound: 1757.1219706
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.1054668

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -342.8887634, 1394.5758057, -314.7524109, 1277.9980469, -1620.8868408, 1709.3281250
1: -420.2510071, 1557.6014404, -385.6337891, 1426.8695068, -1847.1204834, 1943.2349854
2: -480.3132935, 1582.4733887, -441.0567932, 1450.1931152, -1930.5063477, 2023.5300293
3: -682.2117920, 1723.1215820, -626.3041992, 1580.1624756, -2262.3742676, 2349.4257812
4: -805.6596069, 1608.7817383, -740.3444824, 1474.6290283, -2280.2885742, 2349.1262207

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0360244
time: 0.65 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0360244
time: 0.81 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -342.8887634, 1394.5758057, -1747.3286133, 1772.7604980
1: -432.3590393, 1596.3819580, -420.2510071, 1557.6014404, -1989.9603271, 2016.6329346
2: -495.2670288, 1623.0344238, -480.3132935, 1582.4733887, -2077.7404785, 2103.3476562
3: -702.8384399, 1770.8116455, -682.2117920, 1723.1215820, -2425.9599609, 2453.0234375
4: -832.5136719, 1651.0700684, -805.6596069, 1608.7817383, -2441.2954102, 2456.7297363

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0889633, upper bound: 1757.0398435
time: 0.70 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0889633, upper bound: 1757.0398435
time: 0.71 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -342.8887634, 1394.5758057, -352.7528076, 1429.8717041, -1772.7604980, 1747.3286133
1: -420.2510071, 1557.6014404, -432.3590393, 1596.3819580, -2016.6329346, 1989.9603271
2: -480.3132935, 1582.4733887, -495.2670288, 1623.0344238, -2103.3476562, 2077.7404785
3: -682.2117920, 1723.1215820, -702.8384399, 1770.8116455, -2453.0234375, 2425.9599609
4: -805.6596069, 1608.7817383, -832.5136719, 1651.0700684, -2456.7297363, 2441.2954102

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0398435, upper bound: 1757.0889633
time: 0.64 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0398435, upper bound: 1757.0889633
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -338.1303406, 1379.9028320, -450.1930237, 1856.5698242, -2194.6999512, 1830.0955811
1: -414.4423828, 1541.1112061, -550.5508423, 2073.0170898, -2487.4594727, 2091.6618652
2: -473.9914856, 1565.5832520, -638.3911133, 2101.5507812, -2575.5422363, 2203.9743652
3: -673.3363647, 1704.2193604, -902.1766968, 2305.8959961, -2979.2324219, 2606.3959961
4: -795.2887573, 1591.4794922, -1082.5723877, 2138.3459473, -2933.6347656, 2674.0512695

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0398435, upper bound: 1757.0889633
time: 0.73 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0398435, upper bound: 1757.0889633
time: 0.74 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -382.5740051, 1553.1998291, -352.7528076, 1429.8717041, -1812.4456787, 1905.9526367
1: -468.8964844, 1734.6484375, -432.3590393, 1596.3819580, -2065.2783203, 2167.0070801
2: -536.6746216, 1763.2015381, -495.2670288, 1623.0344238, -2159.7089844, 2258.4680176
3: -761.9492188, 1921.3122559, -702.8384399, 1770.8116455, -2532.7604980, 2624.1501465
4: -901.3515625, 1793.1429443, -832.5136719, 1651.0700684, -2552.4216309, 2625.6567383

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1053946, upper bound: 1757.1054186
time: 0.73 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1053946, upper bound: 1757.1054194
time: 0.78 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -378.9179382, 1542.3969727, -450.1930237, 1856.5698242, -2235.4875488, 1992.5899658
1: -464.4025879, 1722.5363770, -550.5508423, 2073.0170898, -2537.4196777, 2273.0869141
2: -531.8988037, 1750.9052734, -638.3911133, 2101.5507812, -2633.4497070, 2389.2963867
3: -755.0411987, 1907.4089355, -902.1766968, 2305.8959961, -3060.9372559, 2809.5854492
4: -893.6969604, 1780.4378662, -1082.5723877, 2138.3459473, -3032.0429688, 2863.0097656

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054040, upper bound: 1757.1054668
time: 0.71 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054040, upper bound: 1757.1054668
time: 0.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.73 seconds
IS_B1_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0360244
IS_B1_A1_B1_A2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -1757.0360244, upper bound: 1757.0360244
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -1757.0889633, upper bound: 1757.0398435
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -1757.0889633, upper bound: 1757.0398435
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -1757.0398435, upper bound: 1757.0889633
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -1757.0398435, upper bound: 1757.0889633
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -1757.0398435, upper bound: 1757.0889633
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -1757.0398435, upper bound: 1757.0889633
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -1757.1053946, upper bound: 1757.1054186
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -1757.1053946, upper bound: 1757.1054194
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -1757.1054040, upper bound: 1757.1054668
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -1757.1054040, upper bound: 1757.1054668

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -314.7524109, 1277.9980469, -1630.7508545, 1744.6241455
1: -432.3590393, 1596.3819580, -385.6337891, 1426.8695068, -1859.2283936, 1982.0156250
2: -495.2670288, 1623.0344238, -441.0567932, 1450.1931152, -1945.4600830, 2064.0913086
3: -702.8384399, 1770.8116455, -626.3041992, 1580.1624756, -2283.0007324, 2397.1157227
4: -832.5136719, 1651.0700684, -740.3444824, 1474.6290283, -2307.1425781, 2391.4145508

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0900739, upper bound: 1757.0268933
time: 0.61 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0919476, upper bound: 1757.0268933
time: 0.68 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -393.3390198, 1617.6787109, -1970.4315186, 1823.2106934
1: -432.3590393, 1596.3819580, -481.2498474, 1806.0983887, -2238.4572754, 2077.6318359
2: -495.2670288, 1623.0344238, -557.5061646, 1832.2119141, -2327.4790039, 2180.5405273
3: -702.8384399, 1770.8116455, -787.0507202, 2007.6927490, -2710.5312500, 2557.8623047
4: -832.5136719, 1651.0700684, -943.0745850, 1864.2309570, -2696.7446289, 2594.1442871

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0900739, upper bound: 1757.0268933
time: 0.70 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0919477, upper bound: 1757.0268933
time: 0.62 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -352.7528076, 1429.8717041, -1744.6240234, 1630.7508545
1: -385.6337891, 1426.8695068, -432.3590393, 1596.3819580, -1982.0155029, 1859.2283936
2: -441.0567932, 1450.1931152, -495.2670288, 1623.0344238, -2064.0913086, 1945.4600830
3: -626.3041992, 1580.1624756, -702.8384399, 1770.8116455, -2397.1157227, 2283.0007324
4: -740.3444824, 1474.6290283, -832.5136719, 1651.0700684, -2391.4145508, 2307.1425781

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0268933, upper bound: 1757.0900739
time: 0.77 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0268933, upper bound: 1757.0919477
time: 0.66 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -393.3390198, 1617.6787109, -352.7528076, 1429.8717041, -1823.2106934, 1970.4315186
1: -481.2498474, 1806.0983887, -432.3590393, 1596.3819580, -2077.6318359, 2238.4572754
2: -557.5061646, 1832.2119141, -495.2670288, 1623.0344238, -2180.5405273, 2327.4790039
3: -787.0507202, 2007.6927490, -702.8384399, 1770.8116455, -2557.8623047, 2710.5312500
4: -943.0745850, 1864.2309570, -832.5136719, 1651.0700684, -2594.1442871, 2696.7446289

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0268933, upper bound: 1757.0900739
time: 0.71 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0268933, upper bound: 1757.0919476
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -450.1930237, 1856.5698242, -2171.3222656, 1728.1907959
1: -385.6337891, 1426.8695068, -550.5508423, 2073.0170898, -2458.6503906, 1977.4204102
2: -441.0567932, 1450.1931152, -638.3911133, 2101.5507812, -2542.6076660, 2088.5842285
3: -626.3041992, 1580.1624756, -902.1766968, 2305.8959961, -2932.2001953, 2482.3388672
4: -740.3444824, 1474.6290283, -1082.5723877, 2138.3459473, -2878.6904297, 2557.2004395

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0262503, upper bound: 1757.0819047
time: 0.65 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0237602, upper bound: 1757.0338508
time: 0.64 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -393.3390198, 1617.6787109, -450.1930237, 1856.5698242, -2249.9084473, 2067.8718262
1: -481.2498474, 1806.0983887, -550.5508423, 2073.0170898, -2554.2668457, 2356.6491699
2: -557.5061646, 1832.2119141, -638.3911133, 2101.5507812, -2659.0568848, 2470.6030273
3: -787.0507202, 2007.6927490, -902.1766968, 2305.8959961, -3092.9467773, 2909.8693848
4: -943.0745850, 1864.2309570, -1082.5723877, 2138.3459473, -3081.4201660, 2946.8032227

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0262503, upper bound: 1757.0819047
time: 0.69 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0237602, upper bound: 1757.0338508
time: 0.96 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -352.7528076, 1429.8717041, -1782.6245117, 1782.6245117
1: -432.3590393, 1596.3819580, -432.3590393, 1596.3819580, -2028.7409668, 2028.7409668
2: -495.2670288, 1623.0344238, -495.2670288, 1623.0344238, -2118.3015137, 2118.3015137
3: -702.8384399, 1770.8116455, -702.8384399, 1770.8116455, -2473.6499023, 2473.6499023
4: -832.5136719, 1651.0700684, -832.5136719, 1651.0700684, -2483.5837402, 2483.5837402

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0385753, upper bound: 1757.0977770
time: 0.70 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0385753, upper bound: 1757.0993832
time: 0.74 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -352.7528076, 1429.8717041, -1880.0645752, 2209.3227539
1: -550.5508423, 2073.0170898, -432.3590393, 1596.3819580, -2146.9328613, 2505.3757324
2: -638.3911133, 2101.5507812, -495.2670288, 1623.0344238, -2261.4255371, 2596.8178711
3: -902.1766968, 2305.8959961, -702.8384399, 1770.8116455, -2672.9882812, 3008.7343750
4: -1082.5723877, 2138.3459473, -832.5136719, 1651.0700684, -2733.6418457, 2970.8596191

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0910416, upper bound: 1757.1161638
time: 0.63 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0385753, upper bound: 1757.0993833
time: 0.70 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -450.1930237, 1856.5698242, -2209.3227539, 1880.0646973
1: -432.3590393, 1596.3819580, -550.5508423, 2073.0170898, -2505.3759766, 2146.9328613
2: -495.2670288, 1623.0344238, -638.3911133, 2101.5507812, -2596.8178711, 2261.4255371
3: -702.8384399, 1770.8116455, -902.1766968, 2305.8959961, -3008.7343750, 2672.9882812
4: -832.5136719, 1651.0700684, -1082.5723877, 2138.3459473, -2970.8596191, 2733.6418457

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0349694, upper bound: 1757.0349694
time: 0.67 seconds

## Relational analysis of IS_B2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0349694, upper bound: 1757.0349694
time: 0.69 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -450.1930237, 1856.5698242, -2306.7626953, 2306.7626953
1: -550.5508423, 2073.0170898, -550.5508423, 2073.0170898, -2623.5678711, 2623.5678711
2: -638.3911133, 2101.5507812, -638.3911133, 2101.5507812, -2739.9418945, 2739.9418945
3: -902.1766968, 2305.8959961, -902.1766968, 2305.8959961, -3208.0727539, 3208.0727539
4: -1082.5723877, 2138.3459473, -1082.5723877, 2138.3459473, -3220.9177246, 3220.9177246

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0349694, upper bound: 1757.0349694
time: 0.71 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0349694, upper bound: 1757.0349694
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.50 seconds
IS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0900739, upper bound: 1757.0268933
IS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0919476, upper bound: 1757.0268933
IS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0900739, upper bound: 1757.0268933
IS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0919477, upper bound: 1757.0268933
IS_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0268933, upper bound: 1757.0900739
IS_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0268933, upper bound: 1757.0919477
IS_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0268933, upper bound: 1757.0900739
IS_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0268933, upper bound: 1757.0919476
IS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0262503, upper bound: 1757.0819047
IS_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0237602, upper bound: 1757.0338508
IS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0262503, upper bound: 1757.0819047
IS_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0237602, upper bound: 1757.0338508
IS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0385753, upper bound: 1757.0977770
IS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0385753, upper bound: 1757.0993832
IS_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0910416, upper bound: 1757.1161638
IS_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0385753, upper bound: 1757.0993833
IS_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0349694, upper bound: 1757.0349694
IS_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0349694, upper bound: 1757.0349694
IS_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0349694, upper bound: 1757.0349694
IS_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 0, lower bound: -1757.0349694, upper bound: 1757.0349694

## BFS IS instance: IS_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -323.3041992, 1308.3963623, -306.6542053, 1245.1453857, -1568.4494629, 1615.0505371
1: -395.9123230, 1461.1677246, -375.7231445, 1390.1154785, -1786.0277100, 1836.8907471
2: -455.4548340, 1484.9721680, -429.7876892, 1412.9300537, -1868.3847656, 1914.7598877
3: -643.2018433, 1623.4108887, -610.1465454, 1539.4003906, -2182.6020508, 2233.5573730
4: -764.8485718, 1512.4916992, -721.1600952, 1436.7764893, -2201.6247559, 2233.6518555

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1136587, upper bound: 1757.1198716
time: 0.76 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1149498, upper bound: 1757.1072259
time: 0.69 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1165965, upper bound: 1757.1235149
time: 0.73 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -342.8786621, 1390.2818604, -313.3518066, 1272.5496826, -1615.4282227, 1703.6334229
1: -420.3764038, 1552.1739502, -383.9295349, 1420.7796631, -1841.1558838, 1936.1032715
2: -481.4416199, 1578.3128662, -439.0760803, 1444.0350342, -1925.4766846, 2017.3889160
3: -683.2790527, 1721.8658447, -623.5477905, 1573.2559814, -2256.5349121, 2345.4135742
4: -809.0892334, 1605.8089600, -736.9074707, 1468.3414307, -2277.4304199, 2342.7160645

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1163906, upper bound: 1757.1071072
time: 0.73 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1180082, upper bound: 1757.1236537
time: 0.77 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -323.3041992, 1308.3963623, -386.6142578, 1589.6594238, -1912.9635010, 1695.0104980
1: -395.9123230, 1461.1677246, -473.1093140, 1774.8972168, -2170.8095703, 1934.2769775
2: -455.4548340, 1484.9721680, -547.9685669, 1800.6972656, -2256.1520996, 2032.9406738
3: -643.2018433, 1623.4108887, -773.5438232, 1972.7099609, -2615.9111328, 2396.9545898
4: -764.8485718, 1512.4916992, -926.5656738, 1832.3294678, -2597.1779785, 2439.0568848

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0860270, upper bound: 1757.0264089
time: 0.73 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A1_B2_A1_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0898959, upper bound: 1757.0253666
time: 0.75 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A2

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0893851, upper bound: 1757.0251848
time: 0.64 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -342.8786621, 1390.2818604, -390.5615540, 1606.3709717, -1949.2493896, 1780.8433838
1: -420.3764038, 1552.1739502, -477.8447571, 1793.4454346, -2213.8217773, 2030.0186768
2: -481.4416199, 1578.3128662, -553.6276855, 1819.4096680, -2300.8513184, 2131.9404297
3: -683.2790527, 1721.8658447, -781.5139771, 1993.7733154, -2677.0522461, 2503.3796387
4: -809.0892334, 1605.8089600, -936.6425171, 1851.1152344, -2660.2045898, 2542.4514160

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_B1_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0184165, upper bound: 1756.9240487
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2_B2

### Relational analysis result of IS_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0910065, upper bound: 1757.0254195
time: 0.66 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -306.6542053, 1245.1453857, -323.3041992, 1308.3963623, -1615.0505371, 1568.4494629
1: -375.7231445, 1390.1154785, -395.9123230, 1461.1677246, -1836.8907471, 1786.0277100
2: -429.7876892, 1412.9300537, -455.4548340, 1484.9721680, -1914.7598877, 1868.3847656
3: -610.1465454, 1539.4003906, -643.2018433, 1623.4108887, -2233.5573730, 2182.6020508
4: -721.1600952, 1436.7764893, -764.8485718, 1512.4916992, -2233.6518555, 2201.6247559

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198717, upper bound: 1757.1136587
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1072259, upper bound: 1757.1149498
time: 0.75 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235149, upper bound: 1757.1165965
time: 0.82 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -313.3518066, 1272.5496826, -342.8786621, 1390.2818604, -1703.6334229, 1615.4282227
1: -383.9295349, 1420.7796631, -420.3764038, 1552.1739502, -1936.1032715, 1841.1558838
2: -439.0760803, 1444.0350342, -481.4416199, 1578.3128662, -2017.3889160, 1925.4766846
3: -623.5477905, 1573.2559814, -683.2790527, 1721.8658447, -2345.4135742, 2256.5349121
4: -736.9074707, 1468.3414307, -809.0892334, 1605.8089600, -2342.7160645, 2277.4304199

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1071072, upper bound: 1757.1163906
time: 0.67 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236537, upper bound: 1757.1180082
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -386.6142578, 1589.6594238, -323.3041992, 1308.3963623, -1695.0104980, 1912.9636230
1: -473.1093140, 1774.8972168, -395.9123230, 1461.1677246, -1934.2769775, 2170.8095703
2: -547.9685669, 1800.6972656, -455.4548340, 1484.9721680, -2032.9406738, 2256.1518555
3: -773.5438232, 1972.7099609, -643.2018433, 1623.4108887, -2396.9545898, 2615.9111328
4: -926.5656738, 1832.3294678, -764.8485718, 1512.4916992, -2439.0568848, 2597.1779785

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0264089, upper bound: 1757.0860270
time: 0.76 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0253666, upper bound: 1757.0898959
time: 0.74 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
time: 0.70 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -390.5615540, 1606.3709717, -342.8786621, 1390.2818604, -1780.8433838, 1949.2495117
1: -477.8447571, 1793.4454346, -420.3764038, 1552.1739502, -2030.0186768, 2213.8217773
2: -553.6276855, 1819.4096680, -481.4416199, 1578.3128662, -2131.9404297, 2300.8513184
3: -781.5139771, 1993.7733154, -683.2790527, 1721.8658447, -2503.3796387, 2677.0522461
4: -936.6425171, 1851.1152344, -809.0892334, 1605.8089600, -2542.4514160, 2660.2045898

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1756.9240486, upper bound: 1757.0184165
time: 0.75 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0254195, upper bound: 1757.0910064
time: 0.87 seconds

## BFS IS instance: IS_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -306.6542053, 1245.1453857, -438.1660156, 1804.7978516, -2111.4519043, 1683.3111572
1: -375.7231445, 1390.1154785, -535.2120361, 2015.2172852, -2390.9404297, 1925.3275146
2: -429.7876892, 1412.9300537, -621.6884155, 2042.3232422, -2472.1108398, 2034.6182861
3: -610.1465454, 1539.4003906, -876.6228027, 2243.8620605, -2854.0085449, 2416.0231934
4: -721.1600952, 1436.7764893, -1054.0126953, 2079.3713379, -2800.5312500, 2490.7890625

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0906620, upper bound: 1757.0158060
time: 0.89 seconds

## Relational analysis of IS_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1168853, upper bound: 1757.0842868
time: 0.71 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -386.6142578, 1589.6594238, -438.1660156, 1804.7978516, -2191.4121094, 2027.8254395
1: -473.1093140, 1774.8972168, -535.2120361, 2015.2172852, -2488.3266602, 2310.1093750
2: -547.9685669, 1800.6972656, -621.6884155, 2042.3232422, -2590.2917480, 2422.3857422
3: -773.5438232, 1972.7099609, -876.6228027, 2243.8620605, -3017.4055176, 2849.3322754
4: -926.5656738, 1832.3294678, -1054.0126953, 2079.3713379, -3005.9365234, 2886.3422852

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1756.9318117, upper bound: 1757.0119897
time: 0.93 seconds

## Relational analysis of IS_B2_A1_B2_A2_B1_B2

### Relational analysis result of IS_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0248907, upper bound: 1757.0807274
time: 0.70 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -344.4634705, 1396.3786621, -323.3041992, 1308.3963623, -1652.8598633, 1719.6827393
1: -422.1711731, 1558.9559326, -395.9123230, 1461.1677246, -1883.3387451, 1954.8681641
2: -483.7297363, 1585.0070801, -455.4548340, 1484.9721680, -1968.7017822, 2040.4619141
3: -686.2062378, 1729.1444092, -643.2018433, 1623.4108887, -2309.6171875, 2372.3461914
4: -812.8952026, 1612.3676758, -764.8485718, 1512.4916992, -2325.3869629, 2377.2163086

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1168270
time: 0.84 seconds

## Relational analysis of IS_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1168270
time: 0.85 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -349.3248901, 1416.1810303, -342.8786621, 1390.2818604, -1739.6066895, 1759.0595703
1: -428.2104797, 1581.0932617, -420.3764038, 1552.1739502, -1980.3843994, 2001.4696045
2: -490.4802856, 1607.5646973, -481.4416199, 1578.3128662, -2068.7932129, 2089.0063477
3: -696.0726318, 1753.8930664, -683.2790527, 1721.8658447, -2417.9384766, 2437.1721191
4: -824.4038696, 1635.3635254, -809.0892334, 1605.8089600, -2430.2126465, 2444.4526367

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1182433
time: 0.76 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1182433
time: 0.74 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -438.1660156, 1804.7978516, -344.4634705, 1396.3786621, -1834.5445557, 2149.2612305
1: -535.2120361, 2015.2172852, -422.1711731, 1558.9559326, -2094.1679688, 2437.3884277
2: -621.6884155, 2042.3232422, -483.7297363, 1585.0070801, -2206.6955566, 2526.0524902
3: -876.6228027, 2243.8620605, -686.2062378, 1729.1444092, -2605.7670898, 2930.0683594
4: -1054.0126953, 2079.3713379, -812.8952026, 1612.3676758, -2666.3803711, 2892.2666016

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_A1

### Relational analysis result of IS_B2_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0284030, upper bound: 1757.0957121
time: 0.62 seconds

## Relational analysis of IS_B2_A2_B1_A2_A1_A2

### Relational analysis result of IS_B2_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0897691, upper bound: 1757.1160876
time: 0.90 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -441.2288513, 1821.4442139, -349.3248901, 1416.1810303, -1857.4099121, 2170.7690430
1: -539.5364380, 2033.7397461, -428.2104797, 1581.0932617, -2120.6296387, 2461.9499512
2: -626.0093384, 2061.7189941, -490.4802856, 1607.5646973, -2233.5737305, 2552.1992188
3: -884.4912720, 2262.6135254, -696.0726318, 1753.8930664, -2638.3842773, 2958.6860352
4: -1062.1268311, 2097.8671875, -824.4038696, 1635.3635254, -2697.4902344, 2922.2709961

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0385753, upper bound: 1757.0977770
time: 0.69 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0385753, upper bound: 1757.0993832
time: 0.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.78 seconds
IS_B1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1149498, upper bound: 1757.1072259
IS_B1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1165965, upper bound: 1757.1235149
IS_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1163906, upper bound: 1757.1071072
IS_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1180082, upper bound: 1757.1236537
IS_B1_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0898959, upper bound: 1757.0253666
IS_B1_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0893851, upper bound: 1757.0251848
IS_B1_A2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0184165, upper bound: 1756.9240487
IS_B1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0910065, upper bound: 1757.0254195
IS_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1072259, upper bound: 1757.1149498
IS_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1235149, upper bound: 1757.1165965
IS_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1071072, upper bound: 1757.1163906
IS_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1236537, upper bound: 1757.1180082
IS_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0253666, upper bound: 1757.0898959
IS_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
IS_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 0, lower bound: -1756.9240486, upper bound: 1757.0184165
IS_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0254195, upper bound: 1757.0910064
IS_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0906620, upper bound: 1757.0158060
IS_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1168853, upper bound: 1757.0842868
IS_B2_A1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.78
Output dim: 0, lower bound: -1756.9318117, upper bound: 1757.0119897
IS_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0248907, upper bound: 1757.0807274
IS_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1168270
IS_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1168270
IS_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1182433
IS_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1182433
IS_B2_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0284030, upper bound: 1757.0957121
IS_B2_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0897691, upper bound: 1757.1160876
IS_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0385753, upper bound: 1757.0977770
IS_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 0, lower bound: -1757.0385753, upper bound: 1757.0993832

## BFS IS instance: IS_B1_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -313.7452698, 1267.8602295, -277.1418762, 1118.2989502, -1432.0441895, 1545.0019531
1: -384.1395569, 1415.8363037, -339.0225525, 1248.7222900, -1632.8616943, 1754.8588867
2: -442.1790771, 1439.1220703, -387.4096375, 1269.7637939, -1711.9427490, 1826.5316162
3: -623.9653931, 1573.8909912, -548.9158325, 1382.4422607, -2006.4077148, 2122.8068848
4: -742.7037354, 1465.9791260, -648.6896973, 1291.3804932, -2034.0842285, 2114.6687012

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1117709, upper bound: 1757.1019072
time: 1.04 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1138341, upper bound: 1757.1041367
time: 0.75 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1149057, upper bound: 1757.1039740
time: 0.70 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -323.3041992, 1308.3963623, -305.0940247, 1238.6749268, -1561.9790039, 1613.4903564
1: -395.9123230, 1461.1677246, -373.8193359, 1382.8961182, -1778.8083496, 1834.9869385
2: -455.4548340, 1484.9721680, -427.5950012, 1405.6096191, -1861.0644531, 1912.5670166
3: -643.2018433, 1623.4108887, -607.0204468, 1531.3800049, -2174.5817871, 2230.4311523
4: -764.8485718, 1512.4916992, -717.4152832, 1429.3585205, -2194.2070312, 2229.9069824

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1136587, upper bound: 1757.1198717
time: 0.72 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0994202, upper bound: 1757.1133258
time: 0.69 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0991402, upper bound: 1757.1102560
time: 0.75 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -332.9656982, 1348.2292480, -283.2286072, 1143.0994873, -1476.0650635, 1631.4578857
1: -408.1769409, 1505.1480713, -346.4918213, 1276.5009766, -1684.6779785, 1851.6395264
2: -467.6494446, 1530.7380371, -395.8165894, 1297.9604492, -1765.6098633, 1926.5545654
3: -663.3303223, 1670.3862305, -561.0640869, 1413.0832520, -2076.4130859, 2231.4501953
4: -786.0402832, 1557.3764648, -662.9075928, 1320.0660400, -2106.1064453, 2220.2841797

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1007482, upper bound: 1757.1058455
time: 0.60 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1007482, upper bound: 1757.1071072
time: 0.71 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -342.8786621, 1390.2818604, -311.7458191, 1265.8824463, -1608.7609863, 1702.0277100
1: -420.3764038, 1552.1739502, -381.9712830, 1413.3395996, -1833.7158203, 1934.1451416
2: -481.4416199, 1578.3128662, -436.8215332, 1436.4943848, -1917.9360352, 2015.1342773
3: -683.2790527, 1721.8658447, -620.3269043, 1565.0079346, -2248.2868652, 2342.1926270
4: -809.0892334, 1605.8089600, -733.0708008, 1460.7044678, -2269.7937012, 2338.8796387

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022068, upper bound: 1757.1222021
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022068, upper bound: 1757.1236537
time: 0.84 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -313.2110901, 1266.9296875, -382.1606750, 1570.9385986, -1884.1496582, 1649.0903320
1: -383.5758057, 1414.9373779, -467.5935364, 1753.9797363, -2137.5554199, 1882.5308838
2: -441.5427246, 1437.8415527, -541.7610474, 1779.3916016, -2220.9340820, 1979.6025391
3: -623.2801514, 1572.9804688, -764.5977173, 1949.9187012, -2573.1984863, 2337.5778809
4: -741.8439941, 1464.8200684, -916.2958984, 1810.6400146, -2552.4836426, 2381.1154785

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0893851, upper bound: 1757.0251848
time: 0.95 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0893851, upper bound: 1757.0251848
time: 0.71 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -306.6980896, 1241.9340820, -380.9374390, 1566.1687012, -1872.8666992, 1622.8714600
1: -375.1054382, 1386.6315918, -466.2052612, 1748.7031250, -2123.8085938, 1852.8365479
2: -433.2834167, 1408.8432617, -539.8941040, 1774.1232910, -2207.4067383, 1948.7373047
3: -610.3952026, 1544.7829590, -762.2212524, 1943.4224854, -2553.8173828, 2307.0039062
4: -729.3919067, 1436.5473633, -912.7282715, 1805.3666992, -2534.7585449, 2349.2756348

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0858562, upper bound: 1757.0248431
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0888298, upper bound: 1757.0169486
time: 0.71 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0893851, upper bound: 1757.0251848
time: 0.66 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0755792, upper bound: 1757.0244665
time: 0.74 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -342.8786621, 1390.2818604, -388.5212708, 1597.9190674, -1940.7976074, 1778.8029785
1: -420.3764038, 1552.1739502, -475.3447266, 1784.0214844, -2204.3979492, 2027.5184326
2: -481.4416199, 1578.3128662, -550.7464600, 1809.8331299, -2291.2746582, 2129.0593262
3: -683.2790527, 1721.8658447, -777.4277344, 1983.3560791, -2666.6352539, 2499.2934570
4: -809.0892334, 1605.8089600, -931.8004761, 1841.3804932, -2650.4697266, 2537.6086426

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0908316, upper bound: 1757.0238954
time: 0.68 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0597953, upper bound: 1757.0220381
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -277.1418762, 1118.2989502, -313.7452698, 1267.8602295, -1545.0019531, 1432.0441895
1: -339.0225525, 1248.7222900, -384.1395569, 1415.8363037, -1754.8588867, 1632.8616943
2: -387.4096375, 1269.7637939, -442.1790771, 1439.1220703, -1826.5316162, 1711.9427490
3: -548.9158325, 1382.4422607, -623.9653931, 1573.8909912, -2122.8068848, 2006.4077148
4: -648.6896973, 1291.3804932, -742.7037354, 1465.9791260, -2114.6687012, 2034.0842285

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1019071, upper bound: 1757.1117709
time: 0.69 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1041367, upper bound: 1757.1138341
time: 0.67 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1039740, upper bound: 1757.1149057
time: 0.72 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -305.0940247, 1238.6749268, -323.3041992, 1308.3963623, -1613.4903564, 1561.9790039
1: -373.8193359, 1382.8961182, -395.9123230, 1461.1677246, -1834.9870605, 1778.8083496
2: -427.5950012, 1405.6096191, -455.4548340, 1484.9721680, -1912.5670166, 1861.0644531
3: -607.0204468, 1531.3800049, -643.2018433, 1623.4108887, -2230.4311523, 2174.5817871
4: -717.4152832, 1429.3585205, -764.8485718, 1512.4916992, -2229.9069824, 2194.2070312

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198717, upper bound: 1757.1136587
time: 0.84 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1133258, upper bound: 1757.0994202
time: 0.66 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1102560, upper bound: 1757.0991402
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -283.2286072, 1143.0994873, -332.9656982, 1348.2292480, -1631.4578857, 1476.0650635
1: -346.4918213, 1276.5009766, -408.1769409, 1505.1480713, -1851.6395264, 1684.6779785
2: -395.8165894, 1297.9604492, -467.6494446, 1530.7380371, -1926.5545654, 1765.6098633
3: -561.0640869, 1413.0832520, -663.3303223, 1670.3862305, -2231.4501953, 2076.4130859
4: -662.9075928, 1320.0660400, -786.0402832, 1557.3764648, -2220.2841797, 2106.1062012

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058455, upper bound: 1757.1007482
time: 0.79 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058455, upper bound: 1757.1163906
time: 0.65 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -311.7458191, 1265.8824463, -342.8786621, 1390.2818604, -1702.0277100, 1608.7609863
1: -381.9712830, 1413.3395996, -420.3764038, 1552.1739502, -1934.1451416, 1833.7158203
2: -436.8215332, 1436.4943848, -481.4416199, 1578.3128662, -2015.1343994, 1917.9360352
3: -620.3269043, 1565.0079346, -683.2790527, 1721.8658447, -2342.1926270, 2248.2868652
4: -733.0708008, 1460.7044678, -809.0892334, 1605.8089600, -2338.8796387, 2269.7937012

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1222021, upper bound: 1757.1022068
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1222021, upper bound: 1757.1180082
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -382.1606750, 1570.9385986, -313.2110901, 1266.9296875, -1649.0903320, 1884.1496582
1: -467.5935364, 1753.9797363, -383.5758057, 1414.9373779, -1882.5308838, 2137.5554199
2: -541.7610474, 1779.3916016, -441.5427246, 1437.8415527, -1979.6025391, 2220.9340820
3: -764.5977173, 1949.9187012, -623.2801514, 1572.9804688, -2337.5778809, 2573.1984863
4: -916.2958984, 1810.6400146, -741.8439941, 1464.8200684, -2381.1154785, 2552.4836426

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
time: 0.69 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
time: 0.67 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -380.9374390, 1566.1687012, -306.6980896, 1241.9340820, -1622.8714600, 1872.8668213
1: -466.2052612, 1748.7031250, -375.1054382, 1386.6315918, -1852.8365479, 2123.8085938
2: -539.8941040, 1774.1232910, -433.2834167, 1408.8432617, -1948.7373047, 2207.4067383
3: -762.2212524, 1943.4224854, -610.3952026, 1544.7829590, -2307.0039062, 2553.8173828
4: -912.7282715, 1805.3666992, -729.3919067, 1436.5473633, -2349.2756348, 2534.7585449

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0248431, upper bound: 1757.0858562
time: 0.65 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
time: 0.71 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
time: 0.78 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -388.5212708, 1597.9190674, -342.8786621, 1390.2818604, -1778.8029785, 1940.7976074
1: -475.3447266, 1784.0214844, -420.3764038, 1552.1739502, -2027.5185547, 2204.3979492
2: -550.7464600, 1809.8331299, -481.4416199, 1578.3128662, -2129.0593262, 2291.2746582
3: -777.4277344, 1983.3560791, -683.2790527, 1721.8658447, -2499.2934570, 2666.6352539
4: -931.8004761, 1841.3804932, -809.0892334, 1605.8089600, -2537.6086426, 2650.4697266

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0238954, upper bound: 1757.0908317
time: 0.73 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0220381, upper bound: 1757.0597953
time: 0.80 seconds

## BFS IS instance: IS_B2_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -301.2449341, 1222.9881592, -446.6734009, 1844.8240967, -2145.3178711, 1669.6616211
1: -369.1342468, 1365.3804932, -544.4739380, 2059.4890137, -2426.7495117, 1909.8543701
2: -422.1726685, 1387.8743896, -634.5224609, 2085.7985840, -2507.9711914, 2022.3968506
3: -599.2890015, 1511.7583008, -893.8937378, 2296.5153809, -2895.5258789, 2405.6520996
4: -708.0023804, 1411.3156738, -1079.9039307, 2122.4238281, -2830.4260254, 2491.2197266

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0902671, upper bound: 1757.0153485
time: 0.64 seconds

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0780709, upper bound: 1757.0004148
time: 0.79 seconds

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0906620, upper bound: 1757.0158060
time: 0.81 seconds

## Relational analysis of IS_B2_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0840816, upper bound: 1757.0156474
time: 0.66 seconds

## BFS IS instance: IS_B2_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -306.6542053, 1245.1453857, -436.0105896, 1796.1485596, -2102.8022461, 1681.1558838
1: -375.7231445, 1390.1154785, -532.5853882, 2005.5345459, -2381.2578125, 1922.7009277
2: -429.7876892, 1412.9300537, -618.6630249, 2032.5235596, -2462.3112793, 2031.5930176
3: -610.1465454, 1539.4003906, -872.3453979, 2233.1633301, -2843.3098145, 2411.7458496
4: -721.1600952, 1436.7764893, -1048.9467773, 2069.4079590, -2790.5681152, 2485.7231445

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1164771, upper bound: 1757.0842388
time: 0.91 seconds

## Relational analysis of IS_B2_A1_B2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1088761, upper bound: 1757.0837006
time: 0.81 seconds

## Relational analysis of IS_B2_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1053537, upper bound: 1757.0837006
time: 0.76 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -386.6142578, 1589.6594238, -436.0105896, 1796.1485596, -2182.7626953, 2025.6699219
1: -473.1093140, 1774.8972168, -532.5853882, 2005.5345459, -2478.6437988, 2307.4821777
2: -547.9685669, 1800.6972656, -618.6630249, 2032.5235596, -2580.4921875, 2419.3603516
3: -773.5438232, 1972.7099609, -872.3453979, 2233.1633301, -3006.7070312, 2845.0551758
4: -926.5656738, 1832.3294678, -1048.9467773, 2069.4079590, -2995.9731445, 2881.2763672

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A2_B1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1756.9259078, upper bound: 1757.0165731
time: 0.67 seconds

## Relational analysis of IS_B2_A1_B2_A2_B1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1756.9259078, upper bound: 1757.0807274
time: 0.71 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -323.3041992, 1308.3963623, -323.3041992, 1308.3963623, -1631.7005615, 1631.7005615
1: -395.9123230, 1461.1677246, -395.9123230, 1461.1677246, -1857.0799561, 1857.0799561
2: -455.4548340, 1484.9721680, -455.4548340, 1484.9721680, -1940.4270020, 1940.4270020
3: -643.2018433, 1623.4108887, -643.2018433, 1623.4108887, -2266.6127930, 2266.6127930
4: -764.8485718, 1512.4916992, -764.8485718, 1512.4916992, -2277.3403320, 2277.3403320

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1023562, upper bound: 1757.1153331
time: 0.73 seconds

## Relational analysis of IS_B2_A2_B1_A1_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1168270
time: 0.83 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -342.8786621, 1390.2818604, -323.3041992, 1308.3963623, -1651.2750244, 1713.5859375
1: -420.3764038, 1552.1739502, -395.9123230, 1461.1677246, -1881.5439453, 1948.0863037
2: -481.4416199, 1578.3128662, -455.4548340, 1484.9721680, -1966.4138184, 2033.7675781
3: -683.2790527, 1721.8658447, -643.2018433, 1623.4108887, -2306.6899414, 2365.0676270
4: -809.0892334, 1605.8089600, -764.8485718, 1512.4916992, -2321.5810547, 2370.6574707

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1023562, upper bound: 1757.1153331
time: 0.66 seconds

## Relational analysis of IS_B2_A2_B1_A1_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1168270
time: 0.78 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -323.2111816, 1308.0167236, -342.8786621, 1390.2818604, -1713.4930420, 1650.8951416
1: -395.7984619, 1460.7447510, -420.3764038, 1552.1739502, -1947.9724121, 1881.1209717
2: -455.3257751, 1484.5421143, -481.4416199, 1578.3128662, -2033.6386719, 1965.9837646
3: -643.0167847, 1622.9438477, -683.2790527, 1721.8658447, -2364.8825684, 2306.2226562
4: -764.6329346, 1512.0555420, -809.0892334, 1605.8089600, -2370.4418945, 2321.1447754

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1155041, upper bound: 1757.1024695
time: 0.80 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1182433
time: 0.72 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -342.8786621, 1390.2818604, -342.8786621, 1390.2818604, -1733.1602783, 1733.1602783
1: -420.3764038, 1552.1739502, -420.3764038, 1552.1739502, -1972.5502930, 1972.5502930
2: -481.4416199, 1578.3128662, -481.4416199, 1578.3128662, -2059.7543945, 2059.7543945
3: -683.2790527, 1721.8658447, -683.2790527, 1721.8658447, -2405.1450195, 2405.1450195
4: -809.0892334, 1605.8089600, -809.0892334, 1605.8089600, -2414.8981934, 2414.8981934

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1155041, upper bound: 1757.1024695
time: 0.78 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1180102
time: 0.96 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -446.6734009, 1844.8240967, -337.9783630, 1369.7224121, -1816.3957520, 2182.0454102
1: -544.4739380, 2059.4890137, -414.2536621, 1529.2435303, -2073.7175293, 2471.8918457
2: -634.5224609, 2085.7985840, -474.5717773, 1554.8906250, -2189.4128418, 2560.3703613
3: -893.8937378, 2296.5153809, -673.0636597, 1695.7991943, -2589.6928711, 2969.5229492
4: -1079.9039307, 2122.4238281, -797.0612183, 1581.7535400, -2661.6574707, 2919.4851074

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0280099, upper bound: 1757.0954803
time: 0.69 seconds

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0284030, upper bound: 1757.0957121
time: 0.68 seconds

## Relational analysis of IS_B2_A2_B1_A2_A1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0284030, upper bound: 1757.0809541
time: 0.80 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -436.0105896, 1796.1485596, -344.4634705, 1396.3786621, -1832.3890381, 2140.6115723
1: -532.5853882, 2005.5345459, -422.1711731, 1558.9559326, -2091.5407715, 2427.7055664
2: -618.6630249, 2032.5235596, -483.7297363, 1585.0070801, -2203.6699219, 2516.2526855
3: -872.3453979, 2233.1633301, -686.2062378, 1729.1444092, -2601.4897461, 2919.3696289
4: -1048.9467773, 2069.4079590, -812.8952026, 1612.3676758, -2661.3144531, 2882.3032227

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0897134, upper bound: 1757.1158477
time: 0.81 seconds

## Relational analysis of IS_B2_A2_B1_A2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0895477, upper bound: 1757.1157901
time: 0.70 seconds

## Relational analysis of IS_B2_A2_B1_A2_A1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0883456, upper bound: 1757.0639378
time: 0.94 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -441.2288513, 1821.4442139, -323.2111816, 1308.0167236, -1749.2456055, 2144.6550293
1: -539.5364380, 2033.7397461, -395.7984619, 1460.7447510, -2000.2810059, 2429.5373535
2: -626.0093384, 2061.7189941, -455.3257751, 1484.5421143, -2110.5515137, 2517.0446777
3: -884.4912720, 2262.6135254, -643.0167847, 1622.9438477, -2507.4348145, 2905.6303711
4: -1062.1268311, 2097.8671875, -764.6329346, 1512.0555420, -2574.1823730, 2862.5000000

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1756.9466213, upper bound: 1757.0336772
time: 0.75 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0373478, upper bound: 1757.0970669
time: 0.72 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -441.2288513, 1821.4442139, -342.8786621, 1390.2818604, -1831.5107422, 2164.3227539
1: -539.5364380, 2033.7397461, -420.3764038, 1552.1739502, -2091.7099609, 2454.1152344
2: -626.0093384, 2061.7189941, -481.4416199, 1578.3128662, -2204.3222656, 2543.1604004
3: -884.4912720, 2262.6135254, -683.2790527, 1721.8658447, -2606.3571777, 2945.8925781
4: -1062.1268311, 2097.8671875, -809.0892334, 1605.8089600, -2667.9357910, 2906.9565430

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1756.9466213, upper bound: 1757.0336772
time: 0.77 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0373478, upper bound: 1757.0970669
time: 0.69 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.96 seconds
IS_B1_A2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1138341, upper bound: 1757.1041367
IS_B1_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1149057, upper bound: 1757.1039740
IS_B1_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0994202, upper bound: 1757.1133258
IS_B1_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0991402, upper bound: 1757.1102560
IS_B1_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1007482, upper bound: 1757.1058455
IS_B1_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1007482, upper bound: 1757.1071072
IS_B1_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1022068, upper bound: 1757.1222021
IS_B1_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1022068, upper bound: 1757.1236537
IS_B1_A2_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0893851, upper bound: 1757.0251848
IS_B1_A2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0893851, upper bound: 1757.0251848
IS_B1_A2_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0893851, upper bound: 1757.0251848
IS_B1_A2_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0755792, upper bound: 1757.0244665
IS_B1_A2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0908316, upper bound: 1757.0238954
IS_B1_A2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0597953, upper bound: 1757.0220381
IS_B2_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1041367, upper bound: 1757.1138341
IS_B2_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1039740, upper bound: 1757.1149057
IS_B2_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1133258, upper bound: 1757.0994202
IS_B2_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1102560, upper bound: 1757.0991402
IS_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1058455, upper bound: 1757.1007482
IS_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1058455, upper bound: 1757.1163906
IS_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1222021, upper bound: 1757.1022068
IS_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1222021, upper bound: 1757.1180082
IS_B2_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
IS_B2_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
IS_B2_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
IS_B2_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0251848, upper bound: 1757.0893851
IS_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0238954, upper bound: 1757.0908317
IS_B2_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0220381, upper bound: 1757.0597953
IS_B2_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0906620, upper bound: 1757.0158060
IS_B2_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0840816, upper bound: 1757.0156474
IS_B2_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1088761, upper bound: 1757.0837006
IS_B2_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1053537, upper bound: 1757.0837006
IS_B2_A1_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.96
Output dim: 0, lower bound: -1756.9259078, upper bound: 1757.0165731
IS_B2_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1756.9259078, upper bound: 1757.0807274
IS_B2_A2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1023562, upper bound: 1757.1153331
IS_B2_A2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1168270
IS_B2_A2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1023562, upper bound: 1757.1153331
IS_B2_A2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1168270
IS_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1155041, upper bound: 1757.1024695
IS_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1182433
IS_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1155041, upper bound: 1757.1024695
IS_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.1168266, upper bound: 1757.1180102
IS_B2_A2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0284030, upper bound: 1757.0957121
IS_B2_A2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0284030, upper bound: 1757.0809541
IS_B2_A2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0895477, upper bound: 1757.1157901
IS_B2_A2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0883456, upper bound: 1757.0639378
IS_B2_A2_B1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.96
Output dim: 0, lower bound: -1756.9466213, upper bound: 1757.0336772
IS_B2_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0373478, upper bound: 1757.0970669
IS_B2_A2_B1_A2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.96
Output dim: 0, lower bound: -1756.9466213, upper bound: 1757.0336772
IS_B2_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 0, lower bound: -1757.0373478, upper bound: 1757.0970669

## BFS IS instance: IS_B1_A2_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -310.7210999, 1256.2452393, -266.1999817, 1076.8791504, -1387.5999756, 1522.4451904
1: -380.4148865, 1402.8244629, -325.4266052, 1202.2532959, -1582.6679688, 1728.2507324
2: -438.0381165, 1425.9169922, -372.6838074, 1222.5895996, -1660.6276855, 1798.6007080
3: -618.0247192, 1559.6414795, -527.3243408, 1331.6804199, -1949.7050781, 2086.9658203
4: -735.8955688, 1452.5827637, -624.6024780, 1243.4389648, -1979.3344727, 2077.1850586

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1092429, upper bound: 1757.0875657
time: 0.69 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1136922, upper bound: 1757.1001809
time: 0.69 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1125815, upper bound: 1757.0954583
time: 0.72 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015965, upper bound: 1757.1031502
time: 0.69 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015965, upper bound: 1757.1041367
time: 0.76 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -313.5375671, 1267.0209961, -276.0760803, 1114.1708984, -1427.7084961, 1543.0970459
1: -383.8836975, 1414.8961182, -337.7267151, 1244.1041260, -1627.9875488, 1752.6228027
2: -441.8856201, 1438.1708984, -385.9060059, 1265.0886230, -1706.9738770, 1824.0766602
3: -623.5532227, 1572.8463135, -546.8178711, 1377.2862549, -2000.8393555, 2119.6640625
4: -742.2140503, 1465.0119629, -646.1967163, 1286.6204834, -2028.8344727, 2111.2087402

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1117045, upper bound: 1757.0993069
time: 0.66 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1146224, upper bound: 1757.0991257
time: 0.73 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1148853, upper bound: 1757.0976902
time: 0.98 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -317.7354431, 1286.0152588, -264.4436340, 1078.3920898, -1396.1275635, 1550.4588623
1: -389.1138916, 1436.1845703, -324.2413940, 1204.3828125, -1593.4967041, 1760.4260254
2: -447.7400818, 1459.5949707, -369.6281738, 1223.6912842, -1671.4312744, 1829.2230225
3: -632.1835938, 1595.8227539, -526.0488281, 1330.0605469, -1962.2440186, 2121.8713379
4: -752.0128174, 1486.6647949, -618.3727417, 1243.5262451, -1995.5386963, 2105.0371094

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0956824, upper bound: 1757.1095870
time: 0.89 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0489637, upper bound: 1757.1059385
time: 0.62 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0886323, upper bound: 1757.1090957
time: 0.72 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -322.4206238, 1304.7048340, -302.1930237, 1226.7327881, -1549.1533203, 1606.8978271
1: -394.8441467, 1457.0349121, -370.2960815, 1369.5234375, -1764.3675537, 1827.3309326
2: -454.2140808, 1480.8051758, -423.5444336, 1392.0905762, -1846.3046875, 1904.3496094
3: -641.4468384, 1618.8427734, -601.2769775, 1516.6424561, -2158.0893555, 2220.1196289
4: -762.7412109, 1508.2675781, -710.5887451, 1415.6564941, -2178.3974609, 2218.8557129

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0955368, upper bound: 1757.1034296
time: 0.70 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0991402, upper bound: 1757.1102560
time: 0.74 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0991402, upper bound: 1757.1102560
time: 0.70 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -311.1098022, 1254.4818115, -283.2286072, 1143.0994873, -1454.2092285, 1537.7104492
1: -380.8917236, 1400.7509766, -346.4918213, 1276.5009766, -1657.3925781, 1747.2424316
2: -435.7752075, 1424.9857178, -395.8165894, 1297.9604492, -1733.7355957, 1820.8022461
3: -617.7591553, 1553.9421387, -561.0640869, 1413.0832520, -2030.8422852, 2115.0063477
4: -731.2055054, 1450.3964844, -662.9075928, 1320.0660400, -2051.2712402, 2113.3039551

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0997356, upper bound: 1757.1027495
time: 0.64 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1007063, upper bound: 1757.1025938
time: 0.69 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -341.1810608, 1383.1949463, -283.2286072, 1143.0994873, -1484.2802734, 1666.4235840
1: -418.2958069, 1544.2584229, -346.4918213, 1276.5009766, -1694.7967529, 1890.7498779
2: -479.0507507, 1570.2958984, -395.8165894, 1297.9604492, -1777.0111084, 1966.1124268
3: -679.8616333, 1713.1204834, -561.0640869, 1413.0832520, -2092.9445801, 2274.1845703
4: -805.0390625, 1597.6628418, -662.9075928, 1320.0660400, -2125.1047363, 2260.5703125

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0997356, upper bound: 1757.1040257
time: 0.72 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1007063, upper bound: 1757.1038639
time: 0.72 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -311.1098022, 1254.4818115, -311.7458191, 1265.8824463, -1576.9921875, 1566.2276611
1: -380.8917236, 1400.7509766, -381.9712830, 1413.3395996, -1794.2312012, 1782.7221680
2: -435.7752075, 1424.9857178, -436.8215332, 1436.4943848, -1872.2695312, 1861.8072510
3: -617.7591553, 1553.9421387, -620.3269043, 1565.0079346, -2182.7670898, 2174.2685547
4: -731.2055054, 1450.3964844, -733.0708008, 1460.7044678, -2191.9099121, 2183.4667969

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0980653, upper bound: 1757.1218840
time: 0.68 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0977432, upper bound: 1757.1190068
time: 0.67 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -341.1810608, 1383.1949463, -311.7458191, 1265.8824463, -1607.0633545, 1694.9407959
1: -418.2958069, 1544.2584229, -381.9712830, 1413.3395996, -1831.6353760, 1926.2296143
2: -479.0507507, 1570.2958984, -436.8215332, 1436.4943848, -1915.5450439, 2007.1174316
3: -679.8616333, 1713.1204834, -620.3269043, 1565.0079346, -2244.8693848, 2333.4472656
4: -805.0390625, 1597.6628418, -733.0708008, 1460.7044678, -2265.7436523, 2330.7329102

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0997356, upper bound: 1757.1236428
time: 0.69 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1007063, upper bound: 1757.1235723
time: 0.77 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -313.2110901, 1266.9296875, -371.0136108, 1524.4447021, -1837.6557617, 1637.9433594
1: -383.5758057, 1414.9373779, -453.7313232, 1701.9355469, -2085.5112305, 1868.6687012
2: -441.5427246, 1437.8415527, -526.2207642, 1726.2058105, -2167.7485352, 1964.0621338
3: -623.2801514, 1572.9804688, -742.1322632, 1893.1644287, -2516.4443359, 2315.1125488
4: -741.8439941, 1464.8200684, -890.6991577, 1756.4783936, -2498.3222656, 2355.5192871

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0774135, upper bound: 1757.0111940
time: 0.79 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0894928, upper bound: 1757.0171302
time: 0.64 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0898959, upper bound: 1757.0253666
time: 0.66 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0763685, upper bound: 1757.0245968
time: 0.76 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -313.2110901, 1266.9296875, -378.4294128, 1560.8175049, -1874.0285645, 1645.3591309
1: -383.5758057, 1414.9373779, -462.3470764, 1742.0063477, -2125.5820312, 1877.2844238
2: -441.5427246, 1437.8415527, -538.2261963, 1766.9385986, -2208.4814453, 1976.0677490
3: -623.2801514, 1572.9804688, -757.8004150, 1940.6715088, -2563.9506836, 2330.7802734
4: -741.8439941, 1464.8200684, -913.3121948, 1798.6665039, -2540.5097656, 2378.1323242

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0894928, upper bound: 1757.0171302
time: 0.74 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0898959, upper bound: 1757.0253666
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0763685, upper bound: 1757.0245968
time: 0.66 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -299.9955139, 1213.6970215, -379.7367554, 1561.2824707, -1861.2779541, 1593.4338379
1: -366.8645325, 1355.0726318, -464.7143555, 1743.2376709, -2110.1015625, 1819.7867432
2: -423.8369141, 1376.8599854, -538.2327271, 1768.5526123, -2192.3889160, 1915.0927734
3: -596.9357300, 1510.2583008, -759.8074951, 1937.4591064, -2534.3940430, 2270.0659180
4: -713.6703491, 1404.0152588, -909.9932251, 1799.6901855, -2513.3603516, 2314.0085449

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0858562, upper bound: 1757.0248431
time: 0.84 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0893851, upper bound: 1757.0251848
time: 0.68 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0893851, upper bound: 1757.0251848
time: 0.74 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -303.8773193, 1228.3267822, -377.9402771, 1553.5765381, -1857.4537354, 1606.2669678
1: -371.5906982, 1371.3087158, -462.5957947, 1734.6583252, -2106.2487793, 1833.9044189
2: -429.3287048, 1393.8560791, -535.5932007, 1759.9372559, -2189.2653809, 1929.4490967
3: -604.6578979, 1529.4365234, -756.2132568, 1927.5780029, -2532.2355957, 2285.6499023
4: -722.6569214, 1422.2009277, -905.2819824, 1790.9023438, -2513.5590820, 2327.4829102

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0755792, upper bound: 1757.0244664
time: 0.73 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0755792, upper bound: 1757.0244665
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -327.7563477, 1327.7172852, -384.1766663, 1579.7045898, -1907.4606934, 1711.8939209
1: -401.9455566, 1482.3604736, -469.9340820, 1763.6610107, -2165.6064453, 1952.2945557
2: -460.5372620, 1507.1990967, -544.6694336, 1789.0643311, -2249.6015625, 2051.8684082
3: -653.3933105, 1645.3671875, -768.6724854, 1961.1318359, -2614.5249023, 2414.0395508
4: -774.1005249, 1533.6435547, -921.7999268, 1820.2277832, -2594.3283691, 2455.4433594

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0597953, upper bound: 1757.0220381
time: 0.76 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0597953, upper bound: 1757.0220381
time: 0.75 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -266.1999817, 1076.8791504, -310.7210999, 1256.2452393, -1522.4451904, 1387.5999756
1: -325.4266052, 1202.2532959, -380.4148865, 1402.8244629, -1728.2507324, 1582.6679688
2: -372.6838074, 1222.5895996, -438.0381165, 1425.9169922, -1798.6007080, 1660.6276855
3: -527.3243408, 1331.6804199, -618.0247192, 1559.6414795, -2086.9658203, 1949.7050781
4: -624.6024780, 1243.4389648, -735.8955688, 1452.5827637, -2077.1848145, 1979.3344727

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875657, upper bound: 1757.1092429
time: 0.66 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1001809, upper bound: 1757.1136922
time: 0.72 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0954583, upper bound: 1757.1125815
time: 0.80 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1031502, upper bound: 1757.1015965
time: 0.91 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1031502, upper bound: 1757.1138341
time: 0.70 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -276.0760803, 1114.1708984, -313.5375671, 1267.0209961, -1543.0970459, 1427.7084961
1: -337.7267151, 1244.1041260, -383.8836975, 1414.8961182, -1752.6228027, 1627.9875488
2: -385.9060059, 1265.0886230, -441.8856201, 1438.1708984, -1824.0766602, 1706.9738770
3: -546.8178711, 1377.2862549, -623.5532227, 1572.8463135, -2119.6640625, 2000.8393555
4: -646.1967163, 1286.6204834, -742.2140503, 1465.0119629, -2111.2087402, 2028.8344727

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0993069, upper bound: 1757.1117045
time: 0.76 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0991257, upper bound: 1757.1146224
time: 0.70 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0976902, upper bound: 1757.1148853
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -264.4436340, 1078.3920898, -317.7354431, 1286.0152588, -1550.4588623, 1396.1275635
1: -324.2413940, 1204.3828125, -389.1138916, 1436.1845703, -1760.4260254, 1593.4967041
2: -369.6281738, 1223.6912842, -447.7400818, 1459.5949707, -1829.2230225, 1671.4312744
3: -526.0488281, 1330.0605469, -632.1835938, 1595.8227539, -2121.8713379, 1962.2440186
4: -618.3727417, 1243.5262451, -752.0128174, 1486.6647949, -2105.0371094, 1995.5386963

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1095870, upper bound: 1757.0956824
time: 0.65 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059386, upper bound: 1757.0489637
time: 0.71 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1090957, upper bound: 1757.0886323
time: 0.85 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -302.1930237, 1226.7327881, -322.4206238, 1304.7048340, -1606.8978271, 1549.1533203
1: -370.2960815, 1369.5234375, -394.8441467, 1457.0349121, -1827.3309326, 1764.3675537
2: -423.5444336, 1392.0905762, -454.2140808, 1480.8051758, -1904.3496094, 1846.3046875
3: -601.2769775, 1516.6424561, -641.4468384, 1618.8427734, -2220.1196289, 2158.0893555
4: -710.5887451, 1415.6564941, -762.7412109, 1508.2675781, -2218.8557129, 2178.3974609

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1034296, upper bound: 1757.0955368
time: 0.67 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A2_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1102560, upper bound: 1757.0991402
time: 0.75 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A2_B2

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1102560, upper bound: 1757.0991402
time: 0.79 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -283.2286072, 1143.0994873, -311.1098022, 1254.4818115, -1537.7104492, 1454.2092285
1: -346.4918213, 1276.5009766, -380.8917236, 1400.7509766, -1747.2424316, 1657.3925781
2: -395.8165894, 1297.9604492, -435.7752075, 1424.9857178, -1820.8022461, 1733.7355957
3: -561.0640869, 1413.0832520, -617.7591553, 1553.9421387, -2115.0063477, 2030.8422852
4: -662.9075928, 1320.0660400, -731.2055054, 1450.3964844, -2113.3037109, 2051.2714844

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022210, upper bound: 1757.0997356
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025938, upper bound: 1757.1007063
time: 0.61 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -283.2286072, 1143.0994873, -341.1810608, 1383.1949463, -1666.4234619, 1484.2801514
1: -346.4918213, 1276.5009766, -418.2958069, 1544.2584229, -1890.7498779, 1694.7967529
2: -395.8165894, 1297.9604492, -479.0507507, 1570.2958984, -1966.1124268, 1777.0111084
3: -561.0640869, 1413.0832520, -679.8616333, 1713.1204834, -2274.1845703, 2092.9445801
4: -662.9075928, 1320.0660400, -805.0390625, 1597.6628418, -2260.5700684, 2125.1044922

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022210, upper bound: 1757.1153056
time: 0.72 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025938, upper bound: 1757.1163473
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -311.7458191, 1265.8824463, -311.1098022, 1254.4818115, -1566.2276611, 1576.9921875
1: -381.9712830, 1413.3395996, -380.8917236, 1400.7509766, -1782.7221680, 1794.2312012
2: -436.8215332, 1436.4943848, -435.7752075, 1424.9857178, -1861.8072510, 1872.2695312
3: -620.3269043, 1565.0079346, -617.7591553, 1553.9421387, -2174.2685547, 2182.7670898
4: -733.0708008, 1460.7044678, -731.2055054, 1450.3964844, -2183.4665527, 2191.9099121

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1218840, upper bound: 1757.0995256
time: 0.70 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1190068, upper bound: 1757.0991343
time: 0.72 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -311.7458191, 1265.8824463, -341.1810608, 1383.1949463, -1694.9407959, 1607.0633545
1: -381.9712830, 1413.3395996, -418.2958069, 1544.2584229, -1926.2296143, 1831.6353760
2: -436.8215332, 1436.4943848, -479.0507507, 1570.2958984, -2007.1174316, 1915.5450439
3: -620.3269043, 1565.0079346, -679.8616333, 1713.1204834, -2333.4472656, 2244.8693848
4: -733.0708008, 1460.7044678, -805.0390625, 1597.6628418, -2330.7329102, 2265.7436523

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1221809, upper bound: 1757.1156644
time: 0.80 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1221571, upper bound: 1757.1180082
time: 0.72 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -371.0136108, 1524.4447021, -313.2110901, 1266.9296875, -1637.9433594, 1837.6557617
1: -453.7313232, 1701.9355469, -383.5758057, 1414.9373779, -1868.6687012, 2085.5112305
2: -526.2207642, 1726.2058105, -441.5427246, 1437.8415527, -1964.0622559, 2167.7485352
3: -742.1322632, 1893.1644287, -623.2801514, 1572.9804688, -2315.1125488, 2516.4443359
4: -890.6991577, 1756.4783936, -741.8439941, 1464.8200684, -2355.5190430, 2498.3222656

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0111940, upper bound: 1757.0774135
time: 0.76 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0171302, upper bound: 1757.0894928
time: 0.67 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0253666, upper bound: 1757.0898959
time: 0.66 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1_B2

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0245968, upper bound: 1757.0763685
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -378.4294128, 1560.8175049, -313.2110901, 1266.9296875, -1645.3591309, 1874.0285645
1: -462.3470764, 1742.0063477, -383.5758057, 1414.9373779, -1877.2844238, 2125.5820312
2: -538.2261963, 1766.9385986, -441.5427246, 1437.8415527, -1976.0677490, 2208.4814453
3: -757.8004150, 1940.6715088, -623.2801514, 1572.9804688, -2330.7802734, 2563.9506836
4: -913.3121948, 1798.6665039, -741.8439941, 1464.8200684, -2378.1323242, 2540.5097656

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0171302, upper bound: 1757.0894928
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0253666, upper bound: 1757.0898959
time: 0.77 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0245968, upper bound: 1757.0763685
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -371.0136108, 1524.4447021, -306.6980896, 1241.9340820, -1612.9473877, 1831.1425781
1: -453.7313232, 1701.9355469, -375.1054382, 1386.6315918, -1840.3629150, 2077.0405273
2: -526.2207642, 1726.2058105, -433.2834167, 1408.8432617, -1935.0638428, 2159.4892578
3: -742.1322632, 1893.1644287, -610.3952026, 1544.7829590, -2286.9150391, 2503.5595703
4: -890.6991577, 1756.4783936, -729.3919067, 1436.5473633, -2327.2465820, 2485.8703613

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=2002.608154296875
rel_dist={0: [-1757.1264512206008, 1757.1264512206008]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1253616, upper bound: 1757.1254498
time: 0.65 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254498, upper bound: 1757.1254498
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.50 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 0, lower bound: -1757.1253616, upper bound: 1757.1254498
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 0, lower bound: -1757.1254498, upper bound: 1757.1254498

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -378.6722107, 1536.7451172, -352.9316101, 1436.0976562, -1814.7696533, 1889.6766357
1: -464.1677856, 1716.4156494, -432.6250916, 1604.0590820, -2068.2265625, 2149.0400391
2: -530.8131104, 1744.9143066, -494.3493042, 1629.8876953, -2160.7006836, 2239.2626953
3: -753.7218628, 1900.4467773, -702.1618652, 1774.0732422, -2527.7951660, 2602.6083984
4: -891.1237793, 1774.4365234, -828.9537354, 1656.8021240, -2547.9252930, 2603.3901367

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1253616, upper bound: 1757.1253616
time: 0.80 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1253616, upper bound: 1757.1254498
time: 0.66 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -394.0172424, 1599.8358154, -392.9946289, 1595.7901611, -1989.8073730, 1992.8304443
1: -482.9465027, 1786.8625488, -481.7033691, 1782.3461914, -2265.2927246, 2268.5659180
2: -552.5497437, 1816.5301514, -551.1360474, 1811.9309082, -2364.4802246, 2367.6660156
3: -784.5815430, 1978.7032471, -782.5770874, 1973.6967773, -2758.2783203, 2761.2802734
4: -927.6616211, 1847.1184082, -925.2904663, 1842.4506836, -2770.1123047, 2772.4089355

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254498, upper bound: 1757.1253616
time: 0.72 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254498, upper bound: 1757.1254498
time: 0.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.70 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 0, lower bound: -1757.1253616, upper bound: 1757.1253616
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 0, lower bound: -1757.1253616, upper bound: 1757.1254498
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 0, lower bound: -1757.1254498, upper bound: 1757.1253616
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 0, lower bound: -1757.1254498, upper bound: 1757.1254498

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -352.9316101, 1436.0976562, -352.9316101, 1436.0976562, -1789.0291748, 1789.0291748
1: -432.6250916, 1604.0590820, -432.6250916, 1604.0590820, -2036.6842041, 2036.6842041
2: -494.3493042, 1629.8876953, -494.3493042, 1629.8876953, -2124.2368164, 2124.2368164
3: -702.1618652, 1774.0732422, -702.1618652, 1774.0732422, -2476.2351074, 2476.2351074
4: -828.9537354, 1656.8021240, -828.9537354, 1656.8021240, -2485.7553711, 2485.7553711

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0746033, upper bound: 1757.0378255
time: 0.67 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.0355282
time: 0.88 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -392.9946289, 1595.7901611, -352.9316101, 1436.0976562, -1829.0922852, 1948.7218018
1: -481.7033691, 1782.3461914, -432.6250916, 1604.0590820, -2085.7622070, 2214.9709473
2: -551.1360474, 1811.9309082, -494.3493042, 1629.8876953, -2181.0236816, 2306.2795410
3: -782.5770874, 1973.6967773, -702.1618652, 1774.0732422, -2556.6503906, 2675.8586426
4: -925.2904663, 1842.4506836, -828.9537354, 1656.8021240, -2582.0925293, 2671.4042969

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0746033, upper bound: 1757.0378285
time: 0.65 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.0377397
time: 0.64 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -352.9316101, 1436.0976562, -392.9946289, 1595.7901611, -1948.7218018, 1829.0922852
1: -432.6250916, 1604.0590820, -481.7033691, 1782.3461914, -2214.9707031, 2085.7624512
2: -494.3493042, 1629.8876953, -551.1360474, 1811.9309082, -2306.2792969, 2181.0236816
3: -702.1618652, 1774.0732422, -782.5770874, 1973.6967773, -2675.8586426, 2556.6503906
4: -828.9537354, 1656.8021240, -925.2904663, 1842.4506836, -2671.4042969, 2582.0925293

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0378255, upper bound: 1757.0746033
time: 0.79 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.0672346
time: 0.65 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -392.9946289, 1595.7901611, -392.9946289, 1595.7901611, -1988.7847900, 1988.7847900
1: -481.7033691, 1782.3461914, -481.7033691, 1782.3461914, -2264.0493164, 2264.0495605
2: -551.1360474, 1811.9309082, -551.1360474, 1811.9309082, -2363.0666504, 2363.0666504
3: -782.5770874, 1973.6967773, -782.5770874, 1973.6967773, -2756.2739258, 2756.2739258
4: -925.2904663, 1842.4506836, -925.2904663, 1842.4506836, -2767.7412109, 2767.7412109

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0378255, upper bound: 1757.1163285
time: 0.66 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.1045412
time: 0.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.94 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 0, lower bound: -1757.0746033, upper bound: 1757.0378255
IS_B1_A1_A2, status: Status.VERIFIED, split count: 3, time: 2.94
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.0355282
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 0, lower bound: -1757.0746033, upper bound: 1757.0378285
IS_B1_A2_A2, status: Status.VERIFIED, split count: 3, time: 2.94
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.0377397
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 0, lower bound: -1757.0378255, upper bound: 1757.0746033
IS_B2_A1_B2, status: Status.VERIFIED, split count: 3, time: 2.94
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.0672346
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 0, lower bound: -1757.0378255, upper bound: 1757.1163285
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.1045412

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -314.7524109, 1277.9980469, -330.9670410, 1345.1430664, -1659.8953857, 1608.9647217
1: -385.6337891, 1426.8695068, -405.5903931, 1502.2193604, -1887.8529053, 1832.4597168
2: -441.0567932, 1450.1931152, -463.6684875, 1526.3142090, -1967.3709717, 1913.8615723
3: -626.3041992, 1580.1624756, -658.5560913, 1662.5627441, -2288.8669434, 2238.7185059
4: -740.3444824, 1474.6290283, -777.9741211, 1551.8208008, -2292.1652832, 2252.6030273

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.0355282
time: 0.80 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.0355282
time: 0.70 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -330.9670410, 1345.1430664, -1697.8958740, 1760.8385010
1: -432.3590393, 1596.3819580, -405.5903931, 1502.2193604, -1934.5782471, 2001.9721680
2: -495.2670288, 1623.0344238, -463.6684875, 1526.3142090, -2021.5811768, 2086.7028809
3: -702.8384399, 1770.8116455, -658.5560913, 1662.5627441, -2365.4011230, 2429.3676758
4: -832.5136719, 1651.0700684, -777.9741211, 1551.8208008, -2384.3344727, 2429.0439453

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0672346, upper bound: 1757.0377397
time: 0.82 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0672346, upper bound: 1757.0377397
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -330.9670410, 1345.1430664, -352.7528076, 1429.8717041, -1760.8383789, 1697.8958740
1: -405.5903931, 1502.2193604, -432.3590393, 1596.3819580, -2001.9721680, 1934.5782471
2: -463.6684875, 1526.3142090, -495.2670288, 1623.0344238, -2086.7028809, 2021.5811768
3: -658.5560913, 1662.5627441, -702.8384399, 1770.8116455, -2429.3676758, 2365.4011230
4: -777.9741211, 1551.8208008, -832.5136719, 1651.0700684, -2429.0439453, 2384.3344727

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.0672346
time: 0.91 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0377397, upper bound: 1757.0672346
time: 0.63 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -370.3549500, 1502.7481689, -352.7528076, 1429.8717041, -1800.2266846, 1855.5009766
1: -453.9002380, 1678.0783691, -432.3590393, 1596.3819580, -2050.2822266, 2110.4370117
2: -519.7257080, 1705.6947021, -495.2670288, 1623.0344238, -2142.7602539, 2200.9614258
3: -737.7836304, 1859.7982178, -702.8384399, 1770.8116455, -2508.5952148, 2562.6367188
4: -873.2812500, 1734.9594727, -832.5136719, 1651.0700684, -2524.3513184, 2567.4731445

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044646, upper bound: 1757.1044713
time: 0.69 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044646, upper bound: 1757.1044712
time: 0.71 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -370.3179321, 1509.7926025, -450.1930237, 1856.5698242, -2226.8876953, 1959.9853516
1: -453.8044128, 1686.0285645, -550.5508423, 2073.0170898, -2526.8215332, 2236.5793457
2: -520.1462402, 1713.6300049, -638.3911133, 2101.5507812, -2621.6970215, 2352.0209961
3: -738.1925659, 1866.9652100, -902.1766968, 2305.8959961, -3044.0886230, 2769.1418457
4: -874.4108276, 1742.5361328, -1082.5723877, 2138.3459473, -3012.7565918, 2825.1079102

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044646, upper bound: 1757.1045412
time: 0.76 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044646, upper bound: 1757.1045412
time: 0.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.86 seconds
IS_B1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.0355282
IS_B1_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.0355282
IS_B1_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0672346, upper bound: 1757.0377397
IS_B1_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0672346, upper bound: 1757.0377397
IS_B2_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0355282, upper bound: 1757.0672346
IS_B2_A1_B1_A2, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0377397, upper bound: 1757.0672346
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.1044646, upper bound: 1757.1044713
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.1044646, upper bound: 1757.1044712
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.1044646, upper bound: 1757.1045412
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.1044646, upper bound: 1757.1045412

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -352.7528076, 1429.8717041, -1782.6245117, 1782.6245117
1: -432.3590393, 1596.3819580, -432.3590393, 1596.3819580, -2028.7409668, 2028.7409668
2: -495.2670288, 1623.0344238, -495.2670288, 1623.0344238, -2118.3015137, 2118.3015137
3: -702.8384399, 1770.8116455, -702.8384399, 1770.8116455, -2473.6499023, 2473.6499023
4: -832.5136719, 1651.0700684, -832.5136719, 1651.0700684, -2483.5837402, 2483.5837402

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0870092, upper bound: 1757.1094619
time: 0.70 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0733209
time: 0.91 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0750027
time: 0.89 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -450.1000977, 1856.1950684, -352.7528076, 1429.8717041, -1879.9718018, 2208.9477539
1: -550.4389038, 2072.6018066, -432.3590393, 1596.3819580, -2146.8208008, 2504.9604492
2: -638.2589111, 2101.1264648, -495.2670288, 1623.0344238, -2261.2934570, 2596.3933105
3: -901.9924316, 2305.4287109, -702.8384399, 1770.8116455, -2672.8039551, 3008.2670898
4: -1082.3427734, 2137.9162598, -832.5136719, 1651.0700684, -2733.4128418, 2970.4299316

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0694693, upper bound: 1757.0967342
time: 0.71 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0750027
time: 0.80 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -352.7528076, 1429.8717041, -450.1930237, 1856.5698242, -2209.3227539, 1880.0646973
1: -432.3590393, 1596.3819580, -550.5508423, 2073.0170898, -2505.3759766, 2146.9328613
2: -495.2670288, 1623.0344238, -638.3911133, 2101.5507812, -2596.8178711, 2261.4255371
3: -702.8384399, 1770.8116455, -902.1766968, 2305.8959961, -3008.7343750, 2672.9882812
4: -832.5136719, 1651.0700684, -1082.5723877, 2138.3459473, -2970.8596191, 2733.6418457

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0855459, upper bound: 1757.0872514
time: 0.73 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0343889, upper bound: 1757.0343889
time: 0.61 seconds

## Relational analysis of IS_B2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0343889, upper bound: 1757.0343889
time: 0.74 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -450.1930237, 1856.5698242, -450.1930237, 1856.5698242, -2306.7626953, 2306.7626953
1: -550.5508423, 2073.0170898, -550.5508423, 2073.0170898, -2623.5678711, 2623.5678711
2: -638.3911133, 2101.5507812, -638.3911133, 2101.5507812, -2739.9418945, 2739.9418945
3: -902.1766968, 2305.8959961, -902.1766968, 2305.8959961, -3208.0727539, 3208.0727539
4: -1082.5723877, 2138.3459473, -1082.5723877, 2138.3459473, -3220.9177246, 3220.9177246

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B2_A2_A1

### Relational analysis result of IS_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0855459, upper bound: 1757.0872707
time: 0.70 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0343889, upper bound: 1757.0343889
time: 1.00 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0343889, upper bound: 1757.0343889
time: 0.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.43 seconds
IS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0733209
IS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0750027
IS_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 0, lower bound: -1757.0694693, upper bound: 1757.0967342
IS_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0750027
IS_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 5.43
Output dim: 0, lower bound: -1757.0343889, upper bound: 1757.0343889
IS_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 5.43
Output dim: 0, lower bound: -1757.0343889, upper bound: 1757.0343889
IS_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 5.43
Output dim: 0, lower bound: -1757.0343889, upper bound: 1757.0343889
IS_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 5.43
Output dim: 0, lower bound: -1757.0343889, upper bound: 1757.0343889

## BFS IS instance: IS_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -338.9934082, 1374.0775146, -323.3041992, 1308.3963623, -1647.3897705, 1697.3815918
1: -415.4240723, 1534.0322266, -395.9123230, 1461.1677246, -1876.5916748, 1929.9444580
2: -476.1034546, 1559.6955566, -455.4548340, 1484.9721680, -1961.0755615, 2015.1502686
3: -675.1381226, 1701.4429932, -643.2018433, 1623.4108887, -2298.5490723, 2344.6447754
4: -799.9217529, 1586.5866699, -764.8485718, 1512.4916992, -2312.4135742, 2351.4353027

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1163380, upper bound: 1757.1163392
time: 0.73 seconds

## Relational analysis of IS_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1163380, upper bound: 1757.1163392
time: 0.75 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -346.4861145, 1404.7637939, -342.8786621, 1390.2818604, -1736.7679443, 1747.6422119
1: -424.7637634, 1568.3437500, -420.3764038, 1552.1739502, -1976.9376221, 1988.7199707
2: -486.5004578, 1594.6696777, -481.4416199, 1578.3128662, -2064.8132324, 2076.1113281
3: -690.4386597, 1739.7670898, -683.2790527, 1721.8658447, -2412.3044434, 2423.0456543
4: -817.6423340, 1622.2874756, -809.0892334, 1605.8089600, -2423.4511719, 2431.3767090

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1163380, upper bound: 1757.1177762
time: 0.69 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1163380, upper bound: 1757.1177762
time: 1.16 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -438.1200867, 1804.6134033, -338.9934082, 1374.0775146, -1812.1975098, 2143.6069336
1: -535.1567993, 2015.0118408, -415.4240723, 1534.0322266, -2069.1889648, 2430.4357910
2: -621.6227417, 2042.1135254, -476.1034546, 1559.6955566, -2181.3183594, 2518.2170410
3: -876.5314941, 2243.6308594, -675.1381226, 1701.4429932, -2577.9746094, 2918.7690430
4: -1053.8990479, 2079.1586914, -799.9217529, 1586.5866699, -2640.4858398, 2879.0805664

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0732695
time: 0.65 seconds

## Relational analysis of IS_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0750027
time: 0.95 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -441.1360779, 1821.0694580, -346.4861145, 1404.7637939, -1845.8999023, 2167.5551758
1: -539.4248047, 2033.3242188, -424.7637634, 1568.3437500, -2107.7685547, 2458.0876465
2: -625.8768921, 2061.2941895, -486.5004578, 1594.6696777, -2220.5466309, 2547.7946777
3: -884.3080444, 2262.1462402, -690.4386597, 1739.7670898, -2624.0751953, 2952.5849609
4: -1061.8977051, 2097.4377441, -817.6423340, 1622.2874756, -2684.1848145, 2915.0800781

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0732695
time: 0.74 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0750027
time: 0.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.81 seconds
IS_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -1757.1163380, upper bound: 1757.1163392
IS_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -1757.1163380, upper bound: 1757.1163392
IS_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -1757.1163380, upper bound: 1757.1177762
IS_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -1757.1163380, upper bound: 1757.1177762
IS_B2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0732695
IS_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0750027
IS_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0732695
IS_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -1757.0366173, upper bound: 1757.0750027

## BFS IS instance: IS_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -323.3041992, 1308.3963623, -323.3041992, 1308.3963623, -1631.7005615, 1631.7005615
1: -395.9123230, 1461.1677246, -395.9123230, 1461.1677246, -1857.0799561, 1857.0799561
2: -455.4548340, 1484.9721680, -455.4548340, 1484.9721680, -1940.4270020, 1940.4270020
3: -643.2018433, 1623.4108887, -643.2018433, 1623.4108887, -2266.6127930, 2266.6127930
4: -764.8485718, 1512.4916992, -764.8485718, 1512.4916992, -2277.3403320, 2277.3403320

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1017732, upper bound: 1757.1151503
time: 0.70 seconds

## Relational analysis of IS_B2_A2_B1_A1_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1163385, upper bound: 1757.1163392
time: 0.81 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -342.8786621, 1390.2818604, -323.3041992, 1308.3963623, -1651.2750244, 1713.5859375
1: -420.3764038, 1552.1739502, -395.9123230, 1461.1677246, -1881.5439453, 1948.0863037
2: -481.4416199, 1578.3128662, -455.4548340, 1484.9721680, -1966.4138184, 2033.7675781
3: -683.2790527, 1721.8658447, -643.2018433, 1623.4108887, -2306.6899414, 2365.0676270
4: -809.0892334, 1605.8089600, -764.8485718, 1512.4916992, -2321.5810547, 2370.6574707

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1017732, upper bound: 1757.1151503
time: 0.76 seconds

## Relational analysis of IS_B2_A2_B1_A1_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1163385, upper bound: 1757.1163392
time: 0.77 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -323.2111816, 1308.0167236, -342.8786621, 1390.2818604, -1713.4930420, 1650.8951416
1: -395.7984619, 1460.7447510, -420.3764038, 1552.1739502, -1947.9724121, 1881.1209717
2: -455.3257751, 1484.5421143, -481.4416199, 1578.3128662, -2033.6386719, 1965.9837646
3: -643.0167847, 1622.9438477, -683.2790527, 1721.8658447, -2364.8825684, 2306.2226562
4: -764.6329346, 1512.0555420, -809.0892334, 1605.8089600, -2370.4418945, 2321.1447754

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1152249, upper bound: 1757.1017937
time: 0.70 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1163380, upper bound: 1757.1177762
time: 0.78 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -342.8786621, 1390.2818604, -342.8786621, 1390.2818604, -1733.1602783, 1733.1602783
1: -420.3764038, 1552.1739502, -420.3764038, 1552.1739502, -1972.5502930, 1972.5502930
2: -481.4416199, 1578.3128662, -481.4416199, 1578.3128662, -2059.7543945, 2059.7543945
3: -683.2790527, 1721.8658447, -683.2790527, 1721.8658447, -2405.1450195, 2405.1450195
4: -809.0892334, 1605.8089600, -809.0892334, 1605.8089600, -2414.8981934, 2414.8981934

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1152249, upper bound: 1757.1017937
time: 0.59 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1163380, upper bound: 1757.1175220
time: 0.75 seconds

## BFS IS instance: IS_B2_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -438.1200867, 1804.6134033, -323.3041992, 1308.3963623, -1746.5164795, 2127.9172363
1: -535.1567993, 2015.0118408, -395.9123230, 1461.1677246, -1996.3244629, 2410.9240723
2: -621.6227417, 2042.1135254, -455.4548340, 1484.9721680, -2106.5949707, 2497.5683594
3: -876.5314941, 2243.6308594, -643.2018433, 1623.4108887, -2499.9423828, 2886.8327637
4: -1053.8990479, 2079.1586914, -764.8485718, 1512.4916992, -2566.3906250, 2844.0073242

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 20
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 34
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 6

Time for candidate selection: 15.16 seconds

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 10

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 10

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 34

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 34

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 46

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 46

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 19

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 19

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 45
type: A, layer: 5, pos: 4
type: A, layer: 5, pos: 32
type: B, layer: 5, pos: 45
type: B, layer: 5, pos: 4
type: A, layer: 5, pos: 14
type: B, layer: 5, pos: 14
type: B, layer: 5, pos: 32
type: A, layer: 5, pos: 24
type: B, layer: 5, pos: 24
type: A, layer: 5, pos: 36
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 39
type: B, layer: 5, pos: 36
type: B, layer: 5, pos: 39
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 40
type: A, layer: 5, pos: 17
type: A, layer: 5, pos: 37
type: B, layer: 5, pos: 17
type: A, layer: 5, pos: 3
type: B, layer: 5, pos: 40
type: A, layer: 5, pos: 31
type: B, layer: 5, pos: 3
type: B, layer: 5, pos: 37
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 31
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 9
type: A, layer: 5, pos: 27
type: B, layer: 5, pos: 9
type: B, layer: 5, pos: 27
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 33
type: A, layer: 5, pos: 7
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 7
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 41
type: A, layer: 5, pos: 18
type: B, layer: 5, pos: 41
type: B, layer: 5, pos: 18
type: A, layer: 5, pos: 0
type: A, layer: 5, pos: 11
type: A, layer: 5, pos: 48
type: A, layer: 5, pos: 12
type: B, layer: 5, pos: 0
type: B, layer: 5, pos: 12
type: B, layer: 5, pos: 11
type: B, layer: 5, pos: 48
type: A, layer: 5, pos: 26
type: A, layer: 5, pos: 35
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 35
type: A, layer: 5, pos: 47
type: A, layer: 5, pos: 42
type: B, layer: 5, pos: 47
type: B, layer: 5, pos: 42
type: A, layer: 5, pos: 5
type: B, layer: 5, pos: 5
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 28
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 43
type: A, layer: 5, pos: 13
type: B, layer: 5, pos: 43
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 13
type: A, layer: 5, pos: 20
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 22
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 10
type: B, layer: 5, pos: 10
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 29
type: B, layer: 5, pos: 29
type: A, layer: 5, pos: 2
type: A, layer: 5, pos: 38
type: B, layer: 5, pos: 2
type: B, layer: 5, pos: 38
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 15

Time for candidate selection: 46.76 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 45

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 32

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 45

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 14

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 14

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 32

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 24

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 24

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 49

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 39

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 39

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 37

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 37

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 30

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 30

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 27

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 27

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 16

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 16

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 0

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 0

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 35

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 35

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 47

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 47

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 28

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 43

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 43

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 19

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 6

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 6

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 10

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 10

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 34

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 34

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 2

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 38

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 2

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 38

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 1

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 1

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 7
type: A, layer: 7, pos: 28
type: B, layer: 7, pos: 28
type: A, layer: 7, pos: 39
type: B, layer: 7, pos: 39
type: A, layer: 7, pos: 6
type: A, layer: 7, pos: 42
type: B, layer: 7, pos: 6
type: B, layer: 7, pos: 42
type: A, layer: 7, pos: 20
type: A, layer: 7, pos: 48
type: B, layer: 7, pos: 20
type: B, layer: 7, pos: 48
type: A, layer: 7, pos: 43
type: A, layer: 7, pos: 45
type: B, layer: 7, pos: 43
type: B, layer: 7, pos: 45
type: A, layer: 7, pos: 16
type: B, layer: 7, pos: 16
type: A, layer: 7, pos: 15
type: A, layer: 7, pos: 22
type: A, layer: 7, pos: 8
type: B, layer: 7, pos: 8
type: B, layer: 7, pos: 22
type: A, layer: 7, pos: 18
type: B, layer: 7, pos: 15
type: B, layer: 7, pos: 18
type: A, layer: 7, pos: 36
type: A, layer: 7, pos: 37
type: A, layer: 7, pos: 46
type: A, layer: 7, pos: 41
type: A, layer: 7, pos: 5
type: B, layer: 7, pos: 41
type: B, layer: 7, pos: 37
type: B, layer: 7, pos: 5
type: B, layer: 7, pos: 36
type: A, layer: 7, pos: 24
type: B, layer: 7, pos: 24
type: B, layer: 7, pos: 46
type: A, layer: 7, pos: 0
type: A, layer: 7, pos: 35
type: A, layer: 7, pos: 3
type: A, layer: 7, pos: 7
type: A, layer: 7, pos: 2
type: B, layer: 7, pos: 35
type: A, layer: 7, pos: 4
type: B, layer: 7, pos: 3
type: B, layer: 7, pos: 0
type: A, layer: 7, pos: 29
type: B, layer: 7, pos: 7
type: B, layer: 7, pos: 2
type: A, layer: 7, pos: 47
type: B, layer: 7, pos: 4
type: B, layer: 7, pos: 47
type: B, layer: 7, pos: 29
type: A, layer: 7, pos: 38
type: A, layer: 7, pos: 19
type: B, layer: 7, pos: 38
type: A, layer: 7, pos: 49
type: A, layer: 7, pos: 31
type: B, layer: 7, pos: 19
type: B, layer: 7, pos: 31
type: A, layer: 7, pos: 12
type: B, layer: 7, pos: 49
type: A, layer: 7, pos: 34
type: B, layer: 7, pos: 34
type: B, layer: 7, pos: 12
type: A, layer: 7, pos: 25
type: A, layer: 7, pos: 30
type: B, layer: 7, pos: 25
type: B, layer: 7, pos: 30
type: A, layer: 7, pos: 23
type: A, layer: 7, pos: 33
type: B, layer: 7, pos: 23
type: A, layer: 7, pos: 9
type: B, layer: 7, pos: 33
type: A, layer: 7, pos: 21
type: A, layer: 7, pos: 17
type: A, layer: 7, pos: 44
type: B, layer: 7, pos: 9
type: A, layer: 7, pos: 26
type: A, layer: 7, pos: 10
type: B, layer: 7, pos: 21
type: B, layer: 7, pos: 44
type: A, layer: 7, pos: 27
type: B, layer: 7, pos: 17
type: B, layer: 7, pos: 26
type: B, layer: 7, pos: 10
type: B, layer: 7, pos: 27
type: A, layer: 7, pos: 11
type: B, layer: 7, pos: 11
type: A, layer: 7, pos: 32
type: A, layer: 7, pos: 13
type: B, layer: 7, pos: 32
type: B, layer: 7, pos: 13
type: A, layer: 7, pos: 14
type: B, layer: 7, pos: 14
type: B, layer: 7, pos: 1
type: A, layer: 7, pos: 1
type: A, layer: 7, pos: 40
type: B, layer: 7, pos: 40

Time for candidate selection: 86.77 seconds

### Candidate
type: A, layer: 7, pos: 28

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 28

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 39

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 39

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 6

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 6

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 43

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 45

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 43

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 45

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 16

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 16

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 37

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 46

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 37

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 24

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 24

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 46

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 0

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 35

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 2

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 35

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 0

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 2

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 47

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 47

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 38

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 19

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 38

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 49

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 19

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 49

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 34

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 34

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 30

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 30

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 21

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 10

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 21

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 27

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 10

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 27

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 32

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 32

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 14

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 14

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 1

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 1

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 7, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 9
type: A, layer: 9, pos: 16
type: B, layer: 9, pos: 16
type: A, layer: 9, pos: 2
type: B, layer: 9, pos: 2
type: A, layer: 9, pos: 0
type: A, layer: 9, pos: 43
type: A, layer: 9, pos: 8
type: B, layer: 9, pos: 43
type: B, layer: 9, pos: 0
type: B, layer: 9, pos: 8
type: A, layer: 9, pos: 31
type: A, layer: 9, pos: 3
type: A, layer: 9, pos: 36
type: B, layer: 9, pos: 31
type: B, layer: 9, pos: 3
type: B, layer: 9, pos: 36
type: A, layer: 9, pos: 38
type: A, layer: 9, pos: 45
type: B, layer: 9, pos: 38
type: B, layer: 9, pos: 45
type: A, layer: 9, pos: 10
type: A, layer: 9, pos: 37
type: B, layer: 9, pos: 10
type: A, layer: 9, pos: 13
type: A, layer: 9, pos: 11
type: A, layer: 9, pos: 42
type: A, layer: 9, pos: 32
type: A, layer: 9, pos: 41
type: A, layer: 9, pos: 9
type: B, layer: 9, pos: 37
type: A, layer: 9, pos: 35
type: A, layer: 9, pos: 12
type: A, layer: 9, pos: 22
type: B, layer: 9, pos: 11
type: B, layer: 9, pos: 41
type: B, layer: 9, pos: 22
type: B, layer: 9, pos: 13
type: B, layer: 9, pos: 32
type: B, layer: 9, pos: 42
type: B, layer: 9, pos: 35
type: A, layer: 9, pos: 28
type: B, layer: 9, pos: 9
type: A, layer: 9, pos: 7
type: B, layer: 9, pos: 12
type: B, layer: 9, pos: 7
type: A, layer: 9, pos: 39
type: B, layer: 9, pos: 39
type: B, layer: 9, pos: 28
type: A, layer: 9, pos: 29
type: A, layer: 9, pos: 34
type: A, layer: 9, pos: 48
type: B, layer: 9, pos: 34
type: B, layer: 9, pos: 29
type: A, layer: 9, pos: 1
type: B, layer: 9, pos: 48
type: B, layer: 9, pos: 1
type: A, layer: 9, pos: 27
type: B, layer: 9, pos: 27
type: A, layer: 9, pos: 26
type: A, layer: 9, pos: 33
type: A, layer: 9, pos: 44
type: A, layer: 9, pos: 30
type: A, layer: 9, pos: 40
type: B, layer: 9, pos: 33
type: B, layer: 9, pos: 44
type: B, layer: 9, pos: 40
type: A, layer: 9, pos: 47
type: B, layer: 9, pos: 30
type: B, layer: 9, pos: 26
type: A, layer: 9, pos: 18
type: B, layer: 9, pos: 47
type: A, layer: 9, pos: 19
type: B, layer: 9, pos: 18
type: B, layer: 9, pos: 19
type: A, layer: 9, pos: 25
type: A, layer: 9, pos: 15
type: B, layer: 9, pos: 15
type: B, layer: 9, pos: 25
type: A, layer: 9, pos: 14
type: B, layer: 9, pos: 14
type: A, layer: 9, pos: 23
type: A, layer: 9, pos: 20
type: A, layer: 9, pos: 5
type: A, layer: 9, pos: 17
type: B, layer: 9, pos: 5
type: A, layer: 9, pos: 21
type: B, layer: 9, pos: 21
type: B, layer: 9, pos: 20
type: A, layer: 9, pos: 49
type: B, layer: 9, pos: 23
type: A, layer: 9, pos: 6
type: B, layer: 9, pos: 17
type: B, layer: 9, pos: 49
type: B, layer: 9, pos: 6
type: A, layer: 9, pos: 24
type: A, layer: 9, pos: 46
type: B, layer: 9, pos: 24
type: B, layer: 9, pos: 46
type: A, layer: 9, pos: 4
type: B, layer: 9, pos: 4

Time for candidate selection: 129.81 seconds

### Candidate
type: A, layer: 9, pos: 16

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 16

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 2

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 2

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 0

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 43

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 43

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 0

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 38

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 45

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 38

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 45

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 10

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 37

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 10

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 32

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 37

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 35

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 32

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 35

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 28

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 39

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 39

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 28

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 34

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 34

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 1

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 1

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 27

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 27

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 30

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 47

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 30

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 47

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 19

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 19

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 25

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 14

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 14

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 21

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 21

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 49

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 23

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 6

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 49

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 6

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 24

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 46

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 24

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 46

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 9, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 9, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 11
type: A, layer: 11, pos: 41
type: B, layer: 11, pos: 41
type: A, layer: 11, pos: 8
type: A, layer: 11, pos: 24
type: B, layer: 11, pos: 8
type: B, layer: 11, pos: 24
type: A, layer: 11, pos: 29
type: B, layer: 11, pos: 29
type: A, layer: 11, pos: 20
type: A, layer: 11, pos: 33
type: B, layer: 11, pos: 20
type: A, layer: 11, pos: 1
type: B, layer: 11, pos: 33
type: A, layer: 11, pos: 39
type: A, layer: 11, pos: 32
type: B, layer: 11, pos: 1
type: B, layer: 11, pos: 39
type: B, layer: 11, pos: 32
type: A, layer: 11, pos: 11
type: B, layer: 11, pos: 11
type: A, layer: 11, pos: 22
type: A, layer: 11, pos: 12
type: B, layer: 11, pos: 22
type: B, layer: 11, pos: 12
type: A, layer: 11, pos: 13
type: A, layer: 11, pos: 18
type: B, layer: 11, pos: 13
type: B, layer: 11, pos: 18
type: A, layer: 11, pos: 48
type: B, layer: 11, pos: 48
type: A, layer: 11, pos: 34
type: B, layer: 11, pos: 34
type: A, layer: 11, pos: 47
type: A, layer: 11, pos: 45
type: B, layer: 11, pos: 47
type: B, layer: 11, pos: 45
type: A, layer: 11, pos: 4
type: A, layer: 11, pos: 36
type: B, layer: 11, pos: 36
type: A, layer: 11, pos: 10
type: B, layer: 11, pos: 4
type: A, layer: 11, pos: 46
type: B, layer: 11, pos: 10
type: B, layer: 11, pos: 46
type: A, layer: 11, pos: 17
type: A, layer: 11, pos: 27
type: B, layer: 11, pos: 17
type: B, layer: 11, pos: 27
type: A, layer: 11, pos: 42
type: A, layer: 11, pos: 26
type: A, layer: 11, pos: 7
type: B, layer: 11, pos: 26
type: B, layer: 11, pos: 42
type: B, layer: 11, pos: 7
type: A, layer: 11, pos: 5
type: A, layer: 11, pos: 16
type: B, layer: 11, pos: 5
type: B, layer: 11, pos: 16
type: A, layer: 11, pos: 6
type: A, layer: 11, pos: 9
type: B, layer: 11, pos: 6
type: A, layer: 11, pos: 3
type: A, layer: 11, pos: 21
type: A, layer: 11, pos: 35
type: B, layer: 11, pos: 9
type: B, layer: 11, pos: 3
type: B, layer: 11, pos: 21
type: B, layer: 11, pos: 35
type: A, layer: 11, pos: 19
type: A, layer: 11, pos: 28
type: A, layer: 11, pos: 31
type: B, layer: 11, pos: 19
type: A, layer: 11, pos: 40
type: B, layer: 11, pos: 31
type: B, layer: 11, pos: 40
type: B, layer: 11, pos: 28
type: A, layer: 11, pos: 38
type: A, layer: 11, pos: 44
type: A, layer: 11, pos: 15
type: B, layer: 11, pos: 38
type: B, layer: 11, pos: 44
type: B, layer: 11, pos: 15
type: A, layer: 11, pos: 14
type: A, layer: 11, pos: 0
type: B, layer: 11, pos: 14
type: B, layer: 11, pos: 0
type: A, layer: 11, pos: 30
type: B, layer: 11, pos: 30
type: A, layer: 11, pos: 37
type: B, layer: 11, pos: 37
type: A, layer: 11, pos: 23
type: B, layer: 11, pos: 23
type: A, layer: 11, pos: 49
type: A, layer: 11, pos: 2
type: B, layer: 11, pos: 49
type: B, layer: 11, pos: 2
type: A, layer: 11, pos: 43
type: B, layer: 11, pos: 43
type: A, layer: 11, pos: 25
type: B, layer: 11, pos: 25

Time for candidate selection: 172.62 seconds

### Candidate
type: A, layer: 11, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 24

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 8

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 24

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 20

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 1

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 39

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 32

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 1

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 39

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 32

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 11

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 12

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 18

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 34

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 34

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 47

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 45

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 47

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 45

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 36

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 10

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 4

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 46

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 10

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 46

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 27

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 17

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 27

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 26

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 42

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 16

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 5

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 16

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 6

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 6

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 21

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 35

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 9

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 21

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 35

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 19

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 28

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 19

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 31

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 40

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 28

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 38

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 11, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 38

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 11, pos: 15

## Relational analysis of IS_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=2002.608154296875
rel_dist={0: [-1757.1254498169274, 1757.1254498169274]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1132.32 seconds
